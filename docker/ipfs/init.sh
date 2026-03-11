#!/bin/sh
# Kestrel IPFS node initialization
# Runs on first start via container-init.d

# Allow API access from Docker network (default restricts to localhost)
ipfs config --json API.HTTPHeaders.Access-Control-Allow-Origin '["*"]'
ipfs config --json API.HTTPHeaders.Access-Control-Allow-Methods '["PUT", "POST", "GET"]'

# Listen on all interfaces for API (needed for Docker networking)
ipfs config Addresses.API /ip4/0.0.0.0/tcp/5001
ipfs config Addresses.Gateway /ip4/0.0.0.0/tcp/8080

# Reduce resource usage for small deployments
ipfs config --json Swarm.ConnMgr.LowWater 50
ipfs config --json Swarm.ConnMgr.HighWater 200
ipfs config --json Datastore.BloomFilterSize 1048576

# Enable Reprovider (announce content to DHT)
ipfs config Reprovider.Interval "12h"
ipfs config Reprovider.Strategy "all"

echo "Kestrel IPFS node configured"
