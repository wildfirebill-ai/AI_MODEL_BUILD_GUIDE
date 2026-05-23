# Nginx — Web Server, Reverse Proxy & Load Balancer

Nginx is a high-performance HTTP server, reverse proxy, and load balancer known for its event-driven architecture and low resource consumption.

## Key Concepts

| Concept | Description |
|---------|-------------|
| **http block** | Main HTTP configuration context |
| **server block** | Virtual host definition (domain + port) |
| **location block** | URI path matching with routing rules |
| **upstream** | Group of backend servers for load balancing |
| **Reverse Proxy** | Forwards client requests to backend services |
| **SSL Termination** | Decrypts HTTPS traffic before proxying |
| **Rate Limiting** | Controls request rate per IP/user |

## Basic Configuration

```nginx
# nginx.conf
http {
    upstream model_servers {
        least_conn;
        server 127.0.0.1:8001;
        server 127.0.0.1:8002;
        server 127.0.0.1:8003;
    }

    server {
        listen 80;
        server_name api.example.com;

        location /predict/ {
            proxy_pass http://model_servers;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }
    }
}
```

## Load Balancing Methods

```nginx
upstream backend {
    # Default: round-robin
    # server 10.0.0.1:8001;

    # Least connections
    least_conn;

    # IP hash (sticky sessions)
    ip_hash;

    server 10.0.0.1:8001 weight=3;
    server 10.0.0.2:8001 weight=1;
    server 10.0.0.3:8001 backup;
}
```

## SSL Termination

```nginx
server {
    listen 443 ssl;
    server_name api.example.com;

    ssl_certificate     /etc/ssl/certs/example.crt;
    ssl_certificate_key /etc/ssl/private/example.key;

    location / {
        proxy_pass http://model_servers;
    }
}
```

## Rate Limiting

```nginx
http {
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;

    server {
        location /predict/ {
            limit_req zone=api burst=20 nodelay;
            proxy_pass http://model_servers;
        }
    }
}
```

## Health Checks (Nginx Plus)

```nginx
upstream model_servers {
    zone model_servers 64k;
    server 10.0.0.1:8001;
    server 10.0.0.2:8001;
    health_check interval=5s fails=3 passes=2;
}
```

## Integration Patterns

- **Reverse Proxy for Model Serving**: Route `/predict` to model server backends
- **Load Balancing Across Replicas**: Distribute inference requests evenly
- **API Gateway**: SSL termination, auth, rate limiting at edge
- **Static Asset Serving**: Serve model artifacts, HTML dashboards

## Testing

```bash
nginx -t                    # test configuration
nginx -s reload             # reload without downtime
```

## References

- [Nginx Documentation](https://nginx.org/en/docs/)
- [Nginx Admin Guide](https://docs.nginx.com/nginx/admin-guide/)
