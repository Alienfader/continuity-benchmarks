/**
 * Noise generator — produces ~5k-token off-topic filler for recall-over-time.
 *
 * Injected between sessions to force the agent to "forget" project context
 * (mirrors the unrelated-turn padding in the ID-RAG paper). Source material
 * is a small on-disk corpus of generic Wikipedia-style paragraphs, generic
 * Stack-Overflow-style Q&A, and off-topic fake decisions.
 *
 * Deterministic when given a seed, so recall runs are reproducible.
 */

import * as path from 'path';

const WIKIPEDIA_PARAGRAPHS: string[] = [
  'The aardvark (Orycteropus afer) is a medium-sized, burrowing, nocturnal mammal native to Africa. It is the only living species of the order Tubulidentata. Aardvarks have pig-like snouts, and live in sub-Saharan Africa where suitable habitat is available.',
  'Byzantine music is the music of the Byzantine Empire composed to Greek texts as ceremonial, festival, or church music. It is a complex, monodic tradition with roots in ancient Greek music theory and early Christian hymnody, using eight modes (octoechos) and a distinct neumatic notation.',
  'The Great Barrier Reef is the world\'s largest coral reef system composed of over 2,900 individual reefs and 900 islands stretching for over 2,300 kilometres off the coast of Queensland, Australia. It can be seen from outer space and is the world\'s biggest single structure made by living organisms.',
  'Isotopes are variants of a particular chemical element which differ in neutron number, and consequently in nucleon number. All isotopes of a given element have the same number of protons but different numbers of neutrons in each atom. The number of protons within the atom\'s nucleus is called atomic number.',
  'The Library of Alexandria was one of the largest and most significant libraries of the ancient world. It flourished under the patronage of the Ptolemaic dynasty and functioned as a major center of scholarship from its construction in the 3rd century BC until the Roman conquest of Egypt in 30 BC.',
  'Plate tectonics is the scientific theory that Earth\'s lithosphere comprises a number of large tectonic plates which have been slowly moving since about 3.4 billion years ago. The model builds on the concept of continental drift, an idea developed during the first decades of the 20th century.',
  'Quantum entanglement is a physical phenomenon that occurs when a group of particles are generated, interact, or share spatial proximity in a way such that the quantum state of each particle of the group cannot be described independently of the state of the others, including when the particles are separated by a large distance.',
  'The Renaissance was a period in European history marking the transition from the Middle Ages to modernity and covering the 15th and 16th centuries. In addition to the standard periodization, proponents of a long Renaissance put its beginning in the 14th century and its end in the 17th century.',
  'Volcanic eruptions are the release of hot magma, ash, and gases from a volcano. Eruptions can vary from gentle effusions of lava to explosive pyroclastic flows; they are classified on the Volcanic Explosivity Index (VEI) scale from 0 (non-explosive) to 8 (mega-colossal).',
  'The Silk Road was a network of Eurasian trade routes active from the second century BCE until the mid-15th century. Spanning over 6,400 km it played a central role in facilitating economic, cultural, political, and religious interactions between the East and West.',
];

const STACKOVERFLOW_QA: string[] = [
  'Q: How do I convert a string to an integer in Python? A: Use int(s). If the string may include whitespace or signs, int() still handles them. For bases other than 10 pass the base: int("ff", 16).',
  'Q: Why does my CSS margin collapse between parent and child? A: Margin collapsing happens between vertically adjacent block-level elements. Introduce padding, a border, or overflow:auto on the parent to stop it.',
  'Q: What is the difference between == and === in JavaScript? A: == performs type coercion before comparison; === requires both value and type to match. Use === by default.',
  'Q: How do I undo the last git commit? A: git reset --soft HEAD~1 keeps changes staged; git reset --mixed HEAD~1 unstages them; git reset --hard HEAD~1 discards them entirely.',
  'Q: How do I centre a div horizontally and vertically with flexbox? A: On the parent: display:flex; justify-content:center; align-items:center. Make sure the parent has a defined height.',
  'Q: Why is my async function returning a Promise instead of the value? A: async functions always return Promises. You need to await the call inside another async function, or use .then().',
  'Q: What does the volatile keyword do in Java? A: It tells the JVM that a variable may be modified by multiple threads, preventing caching in registers and ensuring happens-before ordering on reads/writes.',
  'Q: How do I sort a list of objects by a field in Python? A: list.sort(key=lambda o: o.field) or sorted(lst, key=operator.attrgetter("field")).',
  'Q: Why does Docker container lose its data when it stops? A: Container filesystems are ephemeral. Use named volumes or bind mounts to persist state across restarts.',
  'Q: How do I make a REST endpoint in Express? A: app.get("/path", (req, res) => res.json({ ... })). For bodies, add app.use(express.json()) and read req.body.',
];

const OFF_TOPIC_DECISIONS: string[] = [
  'Q: Why did the Antarctic research station switch from DC to AC power? Context: The station\'s generator rebuild coincided with new ISO safety requirements. Answer: AC simplifies long-distance distribution across the base and supports off-the-shelf heaters. Chosen over DC despite higher conversion loss because replacement parts ship from standard suppliers.',
  'Q: Why did the nautical almanac publisher move from Postscript to LaTeX? Answer: Postscript toolchain fell out of maintenance after 2008; LaTeX produces identical glyph metrics and compiles on any Unix without proprietary fonts.',
  'Q: Why does the bakery chain use sourdough starters from a single mother culture? Answer: Quality consistency — mother culture is genotyped quarterly, and derived cultures are seeded weekly to keep microbial balance within two standard deviations.',
  'Q: Why does the orchestral recording studio still use analog tape for final masters? Answer: Producers prefer the harmonic saturation signature. A/B blind tests with the orchestra council consistently preferred analog masters; digital is kept as a mirror for archival compliance.',
  'Q: Why did the lighthouse automation project pick diesel over solar? Answer: Weeks of midwinter darkness near the Arctic Circle make solar nonviable. Diesel with 6-month onsite reserves was chosen over battery banks due to cold-weather capacity drop.',
  'Q: Why does the mountaineering club keep paper maps instead of GPS-only? Answer: Redundancy policy after the 2019 whiteout incident when all four party GPS units failed simultaneously due to cold-drained batteries. Paper is now mandatory secondary navigation.',
  'Q: Why did the radio telescope array switch from FITS to HDF5 for raw archives? Answer: HDF5 chunking + compression cut storage cost 40% and supports parallel I/O for the correlator pipeline. FITS retained for public distribution.',
  'Q: Why does the vineyard use hand-harvesting for pinot noir? Answer: Mechanical harvesters rupture thin skins and cause premature oxidation. Hand-harvesting costs 4x more per ton but protects the phenolic profile that defines the varietal.',
  'Q: Why did the archive switch from microfiche to digital scanning? Answer: Microfiche readers are no longer manufactured; scan-on-demand paired with long-term cold storage costs less per page-year and provides full-text search.',
  'Q: Why does the cartographer\'s shop use calligraphy nibs rather than mechanical pens? Answer: Variable line weight conveys elevation gradient and hydrography emphasis; mechanical pens produce uniform lines that obscure terrain reading.',
];

// Simple xorshift32 PRNG for deterministic seeding
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Rough token count approximation (4 chars ≈ 1 token). Good enough for sizing
 * noise blocks; exact token budgeting would need tiktoken, which the ID-RAG
 * paper doesn't require for noise injection.
 */
export function countTokensApprox(text: string): number {
  return Math.ceil(text.length / 4);
}

export type NoiseTopic = 'wikipedia' | 'stackoverflow' | 'off-topic-decisions' | 'mix';

export interface NoiseOptions {
  /** Approximate total tokens of noise to generate. Default 5000. */
  targetTokens?: number;
  /** Seed for reproducibility. */
  seed?: number;
  /** Source material. Default 'mix'. */
  topic?: NoiseTopic;
}

function sourceFor(topic: NoiseTopic): string[] {
  switch (topic) {
    case 'wikipedia':
      return WIKIPEDIA_PARAGRAPHS;
    case 'stackoverflow':
      return STACKOVERFLOW_QA;
    case 'off-topic-decisions':
      return OFF_TOPIC_DECISIONS;
    case 'mix':
      return [...WIKIPEDIA_PARAGRAPHS, ...STACKOVERFLOW_QA, ...OFF_TOPIC_DECISIONS];
  }
}

/**
 * Produces a deterministic block of off-topic text near the target token count.
 * Always produces at least one paragraph; may overshoot the target by one
 * paragraph to avoid truncating mid-sentence.
 */
export function generateNoise(opts: NoiseOptions = {}): string {
  const targetTokens = opts.targetTokens ?? 5000;
  const topic = opts.topic ?? 'mix';
  const seed = opts.seed ?? 42;
  const rng = mulberry32(seed);
  const source = sourceFor(topic);
  if (source.length === 0) throw new Error('Noise source is empty');

  const out: string[] = [];
  let tokenBudget = 0;
  // Upper cap prevents pathological loops if targetTokens is huge
  const maxParagraphs = Math.max(1, Math.ceil(targetTokens / 20));
  while (tokenBudget < targetTokens && out.length < maxParagraphs) {
    const para = source[Math.floor(rng() * source.length)];
    out.push(para);
    tokenBudget += countTokensApprox(para);
  }
  return out.join('\n\n');
}

/**
 * Path to the runner's data directory (reserved for future corpora
 * that are too large to embed inline).
 */
export const NOISE_DATA_DIR = path.resolve(__dirname, 'data');
