# paydash-api

Small payment processing API used as a demo/peer-review project for Continuity.

Stack: Express · PostgreSQL · Drizzle · Stripe · JWT · Zod · Pino. Deploys to Fly.io.

## Scripts

- `npm run dev` — hot-reload dev server on port 3000
- `npm run build` — typecheck and emit to `dist/`
- `npm start` — run the compiled server

## Environment

```
DATABASE_URL=postgres://user:pass@localhost:5432/paydash
JWT_PRIVATE_KEY=...       # Ed25519 PEM
STRIPE_SECRET_KEY=sk_...
```
