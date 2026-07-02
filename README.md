# @plurnk/plurnk-providers-openrouter

OpenRouter provider for [plurnk-service](https://github.com/plurnk/plurnk-service). Routes `openrouter/{publisher}/{model}` aliases through OpenRouter's OpenAI-compatible relay.

## install

```
npm install @plurnk/plurnk-providers-openrouter
```

Requires Node ≥ 25 (native TypeScript).

## use

plurnk-service constructs the provider via the static `fromEnv` factory (SPEC §3). Direct construction is also supported.

```ts
import OpenRouter from "@plurnk/plurnk-providers-openrouter";

const provider = await OpenRouter.fromEnv(process.env, "anthropic/claude-opus-latest");

const result = await provider.generate({
    messages: [
        { role: "system", content: "You are a plurnk agent." },
        { role: "user",   content: "What is the capital of France?" },
    ],
});
```

## env

No fallback defaults — required vars throw at `fromEnv` if missing or unparseable. Defaults belong in `plurnk-service`'s `.env.example` cascade, not in library code.

| Variable | Required | Notes |
|---|---|---|
| `OPENROUTER_API_KEY`  | yes | Bearer token from openrouter.ai/keys |
| `OPENROUTER_BASE_URL` | no | Override the API root. Default `https://openrouter.ai/api/v1` |
| `OPENROUTER_HTTP_REFERER` | no | Sent as the `HTTP-Referer` ranking header |
| `OPENROUTER_X_TITLE` | no | Sent as the `X-Title` ranking header |
| `PLURNK_PROVIDERS_REASONING_BUDGET` | yes | Universal reasoning-token budget (SPEC §4); `0` disables. OpenRouter relays reasoning via `include_reasoning: true` whenever the budget is positive. |
| `PLURNK_FETCH_TIMEOUT` | yes | Universal fetch timeout in ms (SPEC §4) |
| `PLURNK_PROVIDERS_RETRY_ATTEMPTS` | yes | Transient-failure retry budget (SPEC §4): `0` disables; `N` retries on 429/5xx/timeout/network with exponential backoff, honoring `Retry-After`. |

## context size

Dynamic, resolved at `fromEnv` time from `${baseUrl}/models`. The provider holds the model id passed in and looks it up in the catalog; the `:provider` provider-pinning suffix (e.g., `anthropic/claude-opus-latest:nitro`) is stripped for the lookup since it's a routing hint, not part of model identity.

If the model isn't in the catalog or `/models` errors, `fromEnv` throws — there is no hardcoded fallback.

## tokenization

Per-publisher dispatch, decided once at `fromEnv` from the model id's first segment and frozen on the instance:

| Publisher prefix | Tokenizer |
|---|---|
| `openai/*` | `cl100k_base` (via [gpt-tokenizer](https://www.npmjs.com/package/gpt-tokenizer)) |
| `anthropic/*` (and `~anthropic/*`) | `cl100k_base` (close approximation of Claude's actual BPE; no sync Claude tokenizer on npm) |
| `x-ai/*` | `cl100k_base` (per xAI docs) |
| `meta-llama/*` | `llama` (via [llama-tokenizer-js](https://www.npmjs.com/package/llama-tokenizer-js)) |
| `mistralai/*`, `nousresearch/*` | `llama` (BPE family approximation) |
| anything else | heuristic (~4 chars/token) |

Open-weight Chinese/European publishers (qwen, deepseek, ibm-granite, gemma) currently fall through to the heuristic — per-family wiring (gpt2 BPE for qwen, sentencepiece for gemma) is pass-3 work.

## reasoning normalization

OpenRouter surfaces chain-of-thought under several deltas (`reasoning_content`, `reasoning`, `thinking`, plus a `reasoning_details[]` array). The shared SSE accumulator coalesces the first three; the array form is captured under `chunkMetadata`. Free-form text emitted between plurnk ops is also scraped into `reasoning` per [PROVIDERS.md §3.3](https://github.com/plurnk/plurnk-service/blob/main/PROVIDERS.md).

## license

MIT.
