// An independent protocol peer using the public Go SDK; no Python wire codecs.
package main

import (
	"context"
	"crypto"
	"crypto/ed25519"
	"crypto/mldsa"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2aclient"
	"github.com/a2aproject/a2a-go/v2/a2acrypto"
	"github.com/a2aproject/a2a-go/v2/a2asrv"
	"github.com/cyberphone/json-canonicalization/go/src/webpki.org/jsoncanonicalizer"
)

type document struct {
	ID      string `json:"id"`
	Methods []struct {
		ID string `json:"id"`
	} `json:"verificationMethod"`
}
type config struct {
	Keys          string   `json:"keys"`
	Replay        string   `json:"replay"`
	Ready         string   `json:"ready"`
	Receipts      string   `json:"receipts"`
	ForeignStable string   `json:"foreignStable"`
	CoreStable    string   `json:"coreStable"`
	Extensions    []string `json:"extensions"`
	ForeignDoc    document `json:"foreignDocument"`
	CoreDoc       document `json:"coreDocument"`
	CoreEd        struct {
		X string `json:"x"`
	} `json:"coreEdPublic"`
	CorePQ      []byte `json:"corePqPublic"`
	CoreCardURL string `json:"coreCardUrl"`
	Protocol    string `json:"protocol"`
}
type keyData struct {
	Ed ed25519.PrivateKey `json:"ed"`
	PQ []byte             `json:"pq"`
}
type signerID struct {
	Alg string `json:"alg"`
	Kid string `json:"kid"`
}
type claims struct {
	Version    int        `json:"version"`
	Protocol   string     `json:"protocol"`
	Extension  string     `json:"extension"`
	Sender     string     `json:"sender"`
	Recipient  string     `json:"recipient"`
	Operation  string     `json:"operation"`
	Digest     string     `json:"request_digest"`
	Timestamp  int64      `json:"timestamp"`
	Nonce      string     `json:"nonce"`
	Extensions []string   `json:"extensions"`
	Signers    []signerID `json:"signers"`
}

func kid(id string) string { return id[strings.LastIndex(id, "#")+1:] }

const proofHeader = "Kestrel-A2A-Sender-Proof"

func must[T any](value T, err error) T {
	if err != nil {
		panic(err)
	}
	return value
}
func check(err error) {
	if err != nil {
		panic(err)
	}
}
func wire(value any) []byte      { return must(json.Marshal(value)) }
func canonical(value any) []byte { return must(jsoncanonicalizer.Transform(wire(value))) }
func digest(value any) string    { v := sha256.Sum256(canonical(value)); return hex.EncodeToString(v[:]) }
func output(value any)           { _, err := os.Stdout.Write(wire(value)); check(err) }
func sorted(values []string) []string {
	v := slices.Clone(values)
	slices.Sort(v)
	return slices.Compact(v)
}
func signedURI(c config) string {
	for _, uri := range c.Extensions {
		if strings.HasSuffix(uri, "#signed-sender-v1") {
			return uri
		}
	}
	panic("missing signed extension")
}
func proof(c config, k keyData, operation string, input any, recipient string, extensions []string) string {
	nonce := make([]byte, 16)
	_, err := rand.Read(nonce)
	check(err)
	value := claims{1, "1.0", signedURI(c), c.ForeignDoc.ID, recipient, operation, digest(input), time.Now().Unix(), hex.EncodeToString(nonce), sorted(extensions), []signerID{{"ed25519", kid(c.ForeignDoc.Methods[0].ID)}, {"ml-dsa-65", kid(c.ForeignDoc.Methods[1].ID)}}}
	payload := canonical(value)
	pq := must(mldsa.NewPrivateKey(mldsa.MLDSA65(), k.PQ))
	pqSig := must(pq.Sign(rand.Reader, payload, crypto.Hash(0)))
	return base64.RawURLEncoding.EncodeToString(payload) + "." + base64.RawURLEncoding.EncodeToString(ed25519.Sign(k.Ed, payload)) + "." + base64.RawURLEncoding.EncodeToString(pqSig)
}

type clientProof struct {
	c config
	k keyData
}

func (p clientProof) Before(ctx context.Context, r *a2aclient.Request) (context.Context, any, error) {
	matched := false
	for _, endpoint := range r.Card.SupportedInterfaces {
		if endpoint.URL == r.BaseURL && string(endpoint.ProtocolBinding) == p.c.Protocol {
			matched = true
		}
	}
	if !matched {
		return ctx, nil, fmt.Errorf("SDK did not select requested transport %s at %s", p.c.Protocol, r.BaseURL)
	}
	if r.ServiceParams == nil {
		r.ServiceParams = make(a2aclient.ServiceParams)
	}
	r.ServiceParams.Append(proofHeader, proof(p.c, p.k, r.Method, r.Payload, p.c.CoreStable, p.c.Extensions))
	r.ServiceParams.Append("A2A-Extensions", strings.Join(p.c.Extensions, ","))
	r.ServiceParams.Append("A2A-Version", "1.0")
	return ctx, nil, nil
}
func (p clientProof) After(context.Context, *a2aclient.Response) error { return nil }

type trustedKeys struct{ c config }

func (t trustedKeys) ResolveKey(ctx context.Context, kid, jku string) (crypto.PublicKey, error) {
	if kid != t.c.CoreDoc.Methods[0].ID || jku != "" {
		return nil, errors.New("untrusted card key")
	}
	return ed25519.PublicKey(must(base64.RawURLEncoding.DecodeString(t.c.CoreEd.X))), nil
}

type serverProof struct {
	c        config
	mu       sync.Mutex
	receipts []map[string]string
}

func (p *serverProof) Before(ctx context.Context, cc *a2asrv.CallContext, r *a2asrv.Request) (context.Context, any, error) {
	deny := func(err error) (context.Context, any, error) {
		return ctx, nil, fmt.Errorf("%w: hybrid proof refused: %v", a2a.ErrUnauthenticated, err)
	}
	headers, _ := cc.ServiceParams().Get(proofHeader)
	if len(headers) != 1 {
		return deny(errors.New("missing proof"))
	}
	parts := strings.Split(headers[0], ".")
	if len(parts) != 3 {
		return deny(errors.New("bad segments"))
	}
	payload, err := base64.RawURLEncoding.DecodeString(parts[0])
	if err != nil {
		return deny(err)
	}
	var claim claims
	if err = json.Unmarshal(payload, &claim); err != nil {
		return deny(err)
	}
	expectedSigners := []signerID{{"ed25519", kid(p.c.CoreDoc.Methods[0].ID)}, {"ml-dsa-65", kid(p.c.CoreDoc.Methods[1].ID)}}
	rawExtensions, _ := cc.ServiceParams().Get("A2A-Extensions")
	extensions := []string{}
	for _, line := range rawExtensions {
		for _, item := range strings.Split(line, ",") {
			extensions = append(extensions, strings.TrimSpace(item))
		}
	}
	if string(payload) != string(canonical(claim)) || claim.Version != 1 || claim.Protocol != "1.0" || claim.Extension != signedURI(p.c) || claim.Sender != p.c.CoreDoc.ID || claim.Recipient != p.c.ForeignStable || claim.Operation != cc.Method() || claim.Digest != digest(r.Payload) || time.Now().Unix()-claim.Timestamp > 300 || claim.Timestamp-time.Now().Unix() > 300 || !slices.Equal(claim.Signers, expectedSigners) || !slices.Equal(claim.Extensions, sorted(extensions)) {
		return deny(errors.New("claim mismatch"))
	}
	sig, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		return deny(err)
	}
	if !ed25519.Verify(ed25519.PublicKey(must(base64.RawURLEncoding.DecodeString(p.c.CoreEd.X))), payload, sig) {
		return deny(errors.New("Ed25519 mismatch"))
	}
	pqSig, err := base64.RawURLEncoding.DecodeString(parts[2])
	if err != nil {
		return deny(err)
	}
	pk, err := mldsa.NewPublicKey(mldsa.MLDSA65(), p.c.CorePQ)
	if err != nil {
		return deny(err)
	}
	if err = mldsa.Verify(pk, payload, pqSig, nil); err != nil {
		return deny(err)
	}
	if len(claim.Nonce) != 32 {
		return deny(errors.New("bad nonce"))
	}
	if _, err = hex.DecodeString(claim.Nonce); err != nil {
		return deny(err)
	}
	key := sha256.Sum256([]byte(claim.Sender + "/" + claim.Nonce))
	path := filepath.Join(p.c.Replay, hex.EncodeToString(key[:]))
	fd, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return deny(err)
	}
	_, writeErr := fd.Write(payload)
	syncErr := fd.Sync()
	closeErr := fd.Close()
	if err = errors.Join(writeErr, syncErr, closeErr); err != nil {
		return deny(err)
	}
	cc.User = &a2asrv.User{Name: claim.Sender, Authenticated: true}
	p.mu.Lock()
	p.receipts = append(p.receipts, map[string]string{"operation": claim.Operation, "sender": claim.Sender})
	p.mu.Unlock()
	for _, uri := range extensions {
		cc.Extensions().Activate(&a2a.AgentExtension{URI: uri})
	}
	return ctx, nil, nil
}
func (p *serverProof) After(context.Context, *a2asrv.CallContext, *a2asrv.Response) error { return nil }

func main() {
	var c config
	check(json.Unmarshal(must(os.ReadFile(os.Args[2])), &c))
	mode := os.Args[1]
	if mode == "keys" {
		pub, priv, err := ed25519.GenerateKey(rand.Reader)
		check(err)
		pq := must(mldsa.GenerateKey(mldsa.MLDSA65()))
		check(os.WriteFile(c.Keys, wire(keyData{priv, pq.Bytes()}), 0600))
		output(map[string]any{"edPublic": map[string]string{"kty": "OKP", "crv": "Ed25519", "x": base64.RawURLEncoding.EncodeToString(pub)}, "pqPublic": base64.StdEncoding.EncodeToString(pq.PublicKey().Bytes())})
		return
	}
	var k keyData
	check(json.Unmarshal(must(os.ReadFile(c.Keys)), &k))
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if mode == "client" {
		httpClient := &http.Client{Timeout: 10 * time.Second, Transport: &http.Transport{}, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
		request := must(http.NewRequestWithContext(ctx, http.MethodGet, c.CoreCardURL, nil))
		response := must(httpClient.Do(request))
		raw := must(io.ReadAll(response.Body))
		check(response.Body.Close())
		if response.StatusCode != 200 {
			panic(string(raw))
		}
		var card a2a.AgentCard
		check(json.Unmarshal(raw, &card))
		verifier := a2acrypto.NewVerifier(a2acrypto.VerifierConfig{KeyResolver: trustedKeys{c}})
		if len(card.Signatures) == 0 {
			panic("unsigned card")
		}
		check(verifier.Verify(ctx, raw, &card.Signatures[0]))
		client := must(a2aclient.NewFromCard(ctx, &card, a2aclient.WithConfig(a2aclient.Config{PreferredTransports: []a2a.TransportProtocol{a2a.TransportProtocol(c.Protocol)}}), a2aclient.WithCallInterceptors(clientProof{c, k}), a2aclient.WithJSONRPCTransport(httpClient), a2aclient.WithRESTTransport(httpClient)))
		defer client.Destroy()
		responseTask := must(client.SendMessage(ctx, &a2a.SendMessageRequest{Message: a2a.NewMessage(a2a.MessageRoleUser, a2a.NewTextPart("Go SDK to Kestrel"))}))
		task, ok := responseTask.(*a2a.Task)
		if !ok {
			panic("expected task")
		}
		current := must(client.GetTask(ctx, &a2a.GetTaskRequest{ID: task.ID}))
		for !current.Status.State.Terminal() {
			select {
			case <-ctx.Done():
				panic(fmt.Errorf("task %s did not finish: %w", task.ID, ctx.Err()))
			case <-time.After(10 * time.Millisecond):
				current = must(client.GetTask(ctx, &a2a.GetTaskRequest{ID: task.ID}))
			}
		}
		if current.Status.State != a2a.TaskStateCompleted {
			panic(fmt.Errorf("task %s ended in %s", task.ID, current.Status.State))
		}
		output(map[string]any{"protocol": c.Protocol, "cardVerified": true, "task": current})
		return
	}
	if mode != "server" {
		panic("bad mode")
	}
	check(os.Mkdir(c.Replay, 0700))
	listener := must(net.Listen("tcp", "127.0.0.1:0"))
	base := "http://" + listener.Addr().String()
	extensions := []map[string]any{}
	for _, uri := range c.Extensions {
		params := map[string]any{"version": 1}
		if uri == signedURI(c) {
			params = map[string]any{"version": 1, "recipient": c.ForeignStable, "signing_did": c.ForeignDoc.ID, "header": proofHeader}
		}
		extensions = append(extensions, map[string]any{"uri": uri, "required": strings.HasSuffix(uri, "#signed-sender-v1") || strings.HasSuffix(uri, "#recipient-policy-v1"), "params": params})
	}
	card := map[string]any{"name": "Independent Go SDK peer", "description": "Native A2A interoperability fixture", "version": "1.0.0", "supportedInterfaces": []map[string]string{{"url": base + "/rpc", "protocolBinding": "JSONRPC", "protocolVersion": "1.0"}, {"url": base + "/rest", "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"}}, "capabilities": map[string]any{"streaming": true, "pushNotifications": false, "extensions": extensions}, "defaultInputModes": []string{"text/plain", "application/json"}, "defaultOutputModes": []string{"text/plain", "application/json"}, "skills": []map[string]any{{"id": "interop", "name": "Interop", "description": "Complete a foreign task", "tags": []string{"test"}}}, "securitySchemes": map[string]any{"signedSender": map[string]any{"apiKeySecurityScheme": map[string]string{"location": "header", "name": proofHeader}}}, "securityRequirements": []map[string]any{{"schemes": map[string]any{"signedSender": map[string]any{}}}}}
	signer := must(a2acrypto.NewSigner(a2acrypto.SignerConfig{PrivateKey: k.Ed, KeyID: c.ForeignDoc.Methods[0].ID}))
	signature := must(signer.Sign(ctx, wire(card)))
	card["signatures"] = []*a2a.AgentCardSignature{signature}
	cardBody := wire(card)
	executor := a2asrv.AgentExecutorFunc(func(ctx context.Context, ec *a2asrv.ExecutorContext) iter.Seq2[a2a.Event, error] {
		return func(yield func(a2a.Event, error) bool) {
			task := a2a.NewSubmittedTask(ec, ec.Message)
			task.Status.State = a2a.TaskStateCompleted
			reply := a2a.NewMessage(a2a.MessageRoleAgent, a2a.NewTextPart("Completed by Go SDK"))
			reply.TaskID = task.ID
			reply.ContextID = task.ContextID
			task.Status.Message = reply
			task.History = append(task.History, reply)
			yield(task, nil)
		}
	})
	auth := &serverProof{c: c}
	handler := a2asrv.NewHandler(executor, a2asrv.WithCallInterceptors(auth))
	mux := http.NewServeMux()
	mux.HandleFunc("/.well-known/agent-card.json", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, err := w.Write(cardBody)
		if err != nil {
			panic(err)
		}
	})
	mux.Handle("/rpc", a2asrv.NewJSONRPCHandler(handler))
	mux.Handle("/rest/", http.StripPrefix("/rest", a2asrv.NewRESTHandler(handler)))
	server := &http.Server{Handler: mux, ReadHeaderTimeout: 5 * time.Second}
	done := make(chan error, 1)
	go func() { done <- server.Serve(listener) }()
	check(os.WriteFile(c.Ready, wire(map[string]any{"base": base, "pid": os.Getpid()}), 0600))
	signals := make(chan os.Signal, 1)
	signal.Notify(signals, syscall.SIGTERM)
	<-signals
	deadline, cancel := context.WithTimeout(ctx, 3*time.Second)
	defer cancel()
	check(server.Shutdown(deadline))
	err := <-done
	if !errors.Is(err, http.ErrServerClosed) {
		check(err)
	}
	check(os.WriteFile(c.Receipts, wire(auth.receipts), 0600))
}
