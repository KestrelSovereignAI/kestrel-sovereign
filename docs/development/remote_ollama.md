Nice, that narrows it down a lot.

Given your constraints:

* ✅ Commercial use OK
* ✅ HTTPS
* ✅ Everything in Docker (Docker Desktop at home, plain Docker on Linux/Win)
* ✅ Completely self-hosted (no SaaS, just stuff you run yourself)
* ✅ No home router / firewall port forwarding

The cleanest architecture is:

> **Home Mac ↔ Reverse-tunnel over outbound connection ↔ VPS with public IP + HTTPS**

So your phone talks HTTPS to your **VPS**, and the VPS forwards traffic through a **reverse tunnel** to your **home Ollama** instance. No ports opened on your home router.

---

## 1. High-level architecture

**Components:**

* **Home machine (Mac / later Win/Linux)**

  * Runs **Ollama** (in or out of Docker)
  * Runs **FRP client (`frpc`)** in Docker – maintains an outbound tunnel to your VPS ([GitHub][1])

* **Cloud VPS (any provider, small $5 box)**

  * Runs **FRP server (`frps`)** in Docker – receives tunnel traffic
  * Runs **Caddy** (or Traefik) in Docker – gets Let’s Encrypt certs, terminates HTTPS, reverse-proxies to `frps` ([Reddit][2])

* **Phone app**

  * Just calls `https://ollama.yourdomain.com`
  * No VPN, no port-forwarding, no SaaS

**Licenses / commercial use:**

* **FRP** – Apache 2.0 (commercial-friendly) ([GitHub][1])
* **Caddy** – Apache 2.0
* **Rathole** (alternative to FRP) – Apache 2.0 ([GitHub][3])
* **Ollama Docker image** – official, fine for commercial usage under their terms ([Docker Hub][4])

All FOSS, all self-hosted.

---

## 2. Concrete flow

1. Phone hits:
   `https://ollama.yourdomain.com/v1/chat/completions`

2. **Caddy (VPS)** terminates HTTPS and forwards HTTP to `frps` on an internal port (e.g. `frps:8080`).

3. **frps (VPS)** knows that `ollama.yourdomain.com` belongs to a tunnel defined by `frpc` and forwards the request over the encrypted tunnel.

4. **frpc (home)** receives the request and proxies it to `http://host.docker.internal:11434` (Ollama REST API).

5. Ollama responds → back through frp → Caddy → phone.

No direct inbound connection to your home network, only an outbound tunnel you control.

---

## 3. VPS stack in Docker (Caddy + FRP server)

On your VPS, something like:

**`docker-compose.yml` (VPS)**

```yaml
version: "3.9"

services:
  frps:
    image: fatedier/frps
    container_name: frps
    restart: unless-stopped
    volumes:
      - ./frps.ini:/frps.ini:ro
    command: ["-c", "/frps.ini"]
    networks:
      - backend

  caddy:
    image: caddy:latest
    container_name: caddy
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./Caddyfile:/etc/caddy/Caddyfile:ro
      - ./caddy-data:/data
      - ./caddy-config:/config
    depends_on:
      - frps
    networks:
      - backend

networks:
  backend:
    driver: bridge
```

**`frps.ini` (VPS)** – FRP server config:

```ini
[common]
bind_port = 7000          ; for frpc to connect
vhost_http_port = 8080    ; internal HTTP vhost for Caddy to talk to
token = super_secret_token
```

**`Caddyfile` (VPS)** – HTTPS + reverse proxy into FRP:

```caddy
ollama.yourdomain.com {
    encode gzip
    reverse_proxy frps:8080
}
```

* Caddy automatically fetches Let’s Encrypt certificates for `ollama.yourdomain.com`.
* You only open **80/443 on the VPS**, nothing on your home router.

---

## 4. Home machine stack (Ollama + FRP client) in Docker

### 4.1 Ollama

For **Linux/Windows with Docker** you can use the official image: ([Docker Hub][4])

```yaml
services:
  ollama:
    image: ollama/ollama
    container_name: ollama
    restart: unless-stopped
    volumes:
      - ollama-data:/root/.ollama
    environment:
      - OLLAMA_HOST=0.0.0.0:11434
    ports:
      - "11434:11434"  # only bound to local machine
```

For **macOS specifically**, Ollama’s docs recommend running *outside* Docker Desktop for GPU support, and just letting Docker talk to it over `host.docker.internal:11434`. ([Ollama][5])
But if you’re OK with CPU-only you *can* containerize it; it’s just slower.

### 4.2 FRP client (`frpc`) in Docker

Add this alongside the `ollama` service:

**`docker-compose.yml` (home)**

```yaml
version: "3.9"

services:
  ollama:
    image: ollama/ollama
    container_name: ollama
    restart: unless-stopped
    volumes:
      - ollama-data:/root/.ollama
    environment:
      - OLLAMA_HOST=0.0.0.0:11434
    ports:
      - "11434:11434"
    networks:
      - backend

  frpc:
    image: fatedier/frpc
    container_name: frpc
    restart: unless-stopped
    volumes:
      - ./frpc.ini:/frpc.ini:ro
    command: ["-c", "/frpc.ini"]
    networks:
      - backend

networks:
  backend:
    driver: bridge

volumes:
  ollama-data:
```

**`frpc.ini` (home)** – FRP client config:

```ini
[common]
server_addr = your.vps.ip.or.hostname
server_port = 7000
token = super_secret_token

[ollama]
type = http
local_ip = ollama          ; service name in docker network
local_port = 11434
custom_domains = ollama.yourdomain.com
```

* `server_addr`/`server_port` must match your VPS’s `frps.ini`.
* `token` must match as well.
* `custom_domains` must match what Caddy uses (`ollama.yourdomain.com`).

Bring it up on each side with:

```bash
docker compose up -d
```

(VPS and home machine separately.)

---

## 5. Securing access

Ollama itself doesn’t ship with auth, so for **multi-user / remote use** you’ll probably want **one extra layer**:

* Add **basic auth or OAuth/JWT** at **Caddy** level (Caddy has plugins, or you can stick an auth-gateway service between Caddy and frps), or
* Put a small **API gateway** service in front that:

  * Authenticates user (API keys / JWT)
  * Forwards requests to Ollama
  * Adds rate limiting / logging if you care

Your mobile app then talks to that gateway rather than directly to Ollama.

---

## 6. Alternative: self-hosted VPN overlay (WireGuard + Headscale)

If you *don’t* want the VPS to see HTTP traffic at all, or you’re OK requiring a VPN client on each device, another fully self-hosted option is:

* **Headscale** as the control server (Docker on VPS)
* **WireGuard clients** (Tailscale/Headscale compatible) on:

  * Home machine
  * Phone
  * Any other devices

Then your phone talks directly to `https://<wireguard-ip>:11434` and you can terminate HTTPS on the home box itself. Also entirely FOSS, but it means installing VPN clients on each phone, and your phone app has to operate “inside” that VPN.

---

If you want, I can next:

* Tighten the `docker-compose` files for your exact platform (Mac vs Linux/Win), and
* Sketch a minimal “API gateway” in front of Ollama that does auth + HTTPS termination cleanly for your mobile app.

[1]: https://github.com/fatedier/frp?utm_source=chatgpt.com "fatedier/frp: A fast reverse proxy to help you expose a local ..."
[2]: https://www.reddit.com/r/selfhosted/comments/z2qt0j/anyone_using_fast_reverse_proxy/?utm_source=chatgpt.com "Anyone using Fast Reverse Proxy? : r/selfhosted"
[3]: https://github.com/rathole-org/rathole?utm_source=chatgpt.com "rathole-org/rathole"
[4]: https://hub.docker.com/r/ollama/ollama?utm_source=chatgpt.com "Ollama Docker image"
[5]: https://ollama.com/blog/ollama-is-now-available-as-an-official-docker-image?utm_source=chatgpt.com "Ollama is now available as an official Docker image"
