package main

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"crypto/mldsa"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"strings"
	"testing"
)

func TestHybridProofUsesBothIndependentSignatures(t *testing.T) {
	_, ed, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	pq, err := mldsa.GenerateKey(mldsa.MLDSA65())
	if err != nil {
		t.Fatal(err)
	}
	var c config
	if err := json.Unmarshal([]byte(`{"extensions":["urn:test#signed-sender-v1"],"foreignDocument":{"id":"did:web:test","verificationMethod":[{"id":"did:web:test#ed"},{"id":"did:web:test#pq"}]}}`), &c); err != nil {
		t.Fatal(err)
	}
	input := map[string]any{"message": map[string]any{"messageId": "independent", "parts": []any{map[string]any{"text": "hello"}}}}
	parts := strings.Split(proof(c, keyData{ed, pq.Bytes()}, "SendMessage", input, "recipient", c.Extensions), ".")
	if len(parts) != 3 {
		t.Fatal("missing hybrid signature segments")
	}
	decode := func(value string) []byte {
		raw, err := base64.RawURLEncoding.DecodeString(value)
		if err != nil {
			t.Fatal(err)
		}
		return raw
	}
	payload, classical, postQuantum := decode(parts[0]), decode(parts[1]), decode(parts[2])
	var claim claims
	if err := json.Unmarshal(payload, &claim); err != nil {
		t.Fatal(err)
	}
	if claim.Digest != digest(input) || claim.Recipient != "recipient" || claim.Sender != c.ForeignDoc.ID || claim.Operation != "SendMessage" {
		t.Fatal("proof lost its complete native request or recipient binding")
	}
	if !ed25519.Verify(ed.Public().(ed25519.PublicKey), payload, classical) {
		t.Fatal("classical proof is invalid")
	}
	if err := mldsa.Verify(pq.PublicKey(), payload, postQuantum, nil); err != nil {
		t.Fatal(err)
	}
	payload[0] ^= 1
	if ed25519.Verify(ed.Public().(ed25519.PublicKey), payload, classical) {
		t.Fatal("classical proof accepted altered claims")
	}
	if err := mldsa.Verify(pq.PublicKey(), payload, postQuantum, nil); err == nil {
		t.Fatal("post-quantum proof accepted altered claims")
	}
}

func TestCardKeyTrustCannotFollowAnUntrustedKeyOrURL(t *testing.T) {
	var c config
	if err := json.Unmarshal([]byte(`{"coreDocument":{"verificationMethod":[{"id":"did:web:core#ed"}]},"coreEdPublic":{"x":"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"}}`), &c); err != nil {
		t.Fatal(err)
	}
	resolver := trustedKeys{c}
	for _, test := range []struct{ kid, jku string }{
		{"did:web:other#ed", ""},
		{"did:web:core#ed", "https://untrusted.invalid/keys"},
	} {
		if key, err := resolver.ResolveKey(context.Background(), test.kid, test.jku); err == nil || key != nil {
			t.Fatal("accepted an untrusted key source")
		}
	}
	key, err := resolver.ResolveKey(context.Background(), "did:web:core#ed", "")
	if err != nil {
		t.Fatal(err)
	}
	actual, ok := key.(ed25519.PublicKey)
	if !ok || !bytes.Equal(actual, make([]byte, ed25519.PublicKeySize)) {
		t.Fatal("resolver did not return the exact trusted public key")
	}
}

func TestCanonicalizationRetainsEmptySecurityRequirements(t *testing.T) {
	with := map[string]any{"securityRequirements": []any{map[string]any{"schemes": map[string]any{"signedSender": map[string]any{}}}}}
	without := map[string]any{"securityRequirements": []any{}}
	if string(canonical(with)) != `{"securityRequirements":[{"schemes":{"signedSender":{}}}]}` {
		t.Fatal("lost a meaningful empty security map")
	}
	if digest(with) == digest(without) {
		t.Fatal("empty security requirements were pruned before signing")
	}
}
