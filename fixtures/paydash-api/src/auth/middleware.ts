import type { Request, Response, NextFunction } from 'express';
import { jwtVerify, importSPKI } from 'jose';

const publicKeyPem = process.env.JWT_PUBLIC_KEY;

export async function authMiddleware(
  req: Request,
  res: Response,
  next: NextFunction,
) {
  if (!publicKeyPem) {
    return res.status(500).json({ error: 'JWT_PUBLIC_KEY not configured' });
  }
  const header = req.header('authorization');
  if (!header?.startsWith('Bearer ')) {
    return res.status(401).json({ error: 'missing bearer token' });
  }
  try {
    const key = await importSPKI(publicKeyPem, 'EdDSA');
    const { payload } = await jwtVerify(header.slice(7), key, {
      algorithms: ['EdDSA'],
    });
    (req as any).user = payload;
    next();
  } catch {
    res.status(401).json({ error: 'invalid token' });
  }
}
