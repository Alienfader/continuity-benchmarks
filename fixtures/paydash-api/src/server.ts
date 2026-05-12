import express from 'express';
import pinoHttp from 'pino-http';
import { logger } from './logger.js';
import { authMiddleware } from './auth/middleware.js';
import { paymentsRouter } from './routes/payments.js';

const app = express();

app.use(express.json());
app.use(pinoHttp({ logger }));

app.get('/healthz', (_req, res) => {
  res.json({ ok: true });
});

app.use('/api', authMiddleware);
app.use('/api/payments', paymentsRouter);

const port = Number(process.env.PORT ?? 3000);
app.listen(port, () => {
  logger.info({ port }, 'paydash-api listening');
});
