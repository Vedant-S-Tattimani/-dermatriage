# ── Build Stage ──────────────────────────────────────────────────────────────
FROM node:20-slim AS builder

WORKDIR /build

COPY package*.json ./
RUN npm install

COPY . .
RUN npm run build

# ── Production Stage (Nginx) ─────────────────────────────────────────────────
FROM nginx:stable-alpine

LABEL maintainer="Medical AI Frontend Team"

# Copy build artifacts from builder stage
COPY --from=builder /build/dist /usr/share/nginx/html

# Custom Nginx config for SPA routing and compression
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
