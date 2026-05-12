import { Router } from 'express';
import { z } from 'zod';
import Stripe from 'stripe';
import { logger } from '../logger.js';

const stripe = new Stripe(process.env.STRIPE_SECRET_KEY ?? '');

const CreatePaymentSchema = z.object({
  amountCents: z.number().int().positive(),
  currency: z.string().length(3),
  customerId: z.string().min(1),
});

export const paymentsRouter = Router();

paymentsRouter.post('/', async (req, res) => {
  const parsed = CreatePaymentSchema.safeParse(req.body);
  if (!parsed.success) {
    return res.status(400).json({ errors: parsed.error.flatten() });
  }
  const { amountCents, currency, customerId } = parsed.data;

  const intent = await stripe.paymentIntents.create({
    amount: amountCents,
    currency,
    customer: customerId,
  });

  logger.info({ paymentIntentId: intent.id, customerId }, 'payment intent created');
  res.status(201).json({ clientSecret: intent.client_secret });
});
