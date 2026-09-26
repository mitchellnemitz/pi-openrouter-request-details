# pi-openrouter-request-details

A [Pi](https://github.com/earendil-works/pi) extension that shows which provider
and model OpenRouter actually served for each request.

## What it does

OpenRouter returns the serving provider in a top-level `provider` field on
every chat/completions streaming chunk, but Pi-ai drops it: it keeps the routed
model id (`responseModel`) and the generation id (`responseId`) and ignores the
rest. No extension event carries raw chunks, so the extension reads the field
off the HTTP stream itself: it wraps `globalThis.fetch` (a supported pattern —
Pi core installs undici's fetch as the global and explicitly preserves caller
replacements of it, and the OpenAI client Pi-ai builds per request binds the
global fetch) and, for any `openrouter.ai` streaming response, tees the SSE
body. One branch goes back to Pi-ai untouched; the other is parsed in the
background for the generation id, provider, served model, and usage.

Both OpenRouter stream shapes are parsed. Chat completions puts
id/model/provider/usage at the top level of each chunk; the Responses API
nests them under `response` and omits `provider` entirely — there the provider
only appears in the opt-in `openrouter_metadata` block, so the wrap also sets
`X-OpenRouter-Metadata: enabled` on generation requests to both endpoints
(harmless on completions, where the provider is already top-level).

Every response also gets a one-line header ABOVE it in the transcript - in
place before any of the response streams in:

```
z-ai/glm-5.3 @ Mistral ($1.40 / $4.40 / $0.14)
[the response streams in below]
```

The parenthesized rates are the serving provider's listed prices per million
tokens (in/out/cache-read), from OpenRouter's endpoints catalog
(`/api/v1/models/{author}/{slug}/endpoints` - the only per-provider pricing
source; neither the stream metadata nor the model-level catalog carries it).
The catalog is fetched once at session start for the startup model and
whenever a response is served by a provider the cached catalog does not
list, so a normal session costs one ~40KB fetch per distinct served model.
A provider that lists several priced quantization variants renders a range
(e.g. `$1.40-2.10` in the in-slot); until the catalog is warm the header renders without
the rates and fills them in a moment later. Listed rates are informational -
actual billing can differ (quantization tier actually served, workspace
discounts).

While the response is still streaming, the line fills in live from the tee
(the serving provider and model id ride the first chunks of the stream). A
placeholder entry is appended at `message_start` - before Pi's UI creates the
streaming component, so the header sits directly above the response block -
and a complete copy is appended at `message_end`, which is what persists in
the session file. A rebuilt session (reload, resume) renders only the
complete copy; the placeholder's data stays empty forever.

Each placeholder carries a generation number and its component only
renders while that generation is the one streaming. Without it, every
placeholder component reads the same live state, so during a tool-call
loop the previous request's placeholder would illuminate again with the
current request's header - stacked as an identical duplicate line above
the previous one (Pi renders nothing between them while the turn is
active).

Both blocks render on a `customMessageBg` stripe — the same background fill Pi
uses for its own informational notices (compaction summaries, skill
invocations), lifted 20% so it stands out against the terminal background —
with one cell of padding on every side.

All info-block text renders in the muted grey - the same faded color pi uses
for secondary UI text - so the blocks read as system meta, clearly dimmer
than response text, but not a prompt-submission box either.

Because the header belongs to each response, a tool-call loop shows one
per response. When every request in the turn served the same model @
provider (the common case for a fixed model), consecutive headers show the
same line — that is N responses each with their own header, not one header
rendering N times.

Each turn then gets one compact card in the transcript after it. A turn can
contain many OpenRouter requests — every assistant message in a tool-call loop
is a separate billed request — so the card aggregates the whole turn: usage
and cost are summed across requests. The card defaults to a single metrics
line; expanding (click or ctrl+o) reveals the model/provider row. Expanded
state for the simple case (one request, fixed model id):

```
z-ai/glm-5.3-flash @ BaseTen (x1)

$0.01 | 832 / 24 | 96% cached
```

A multi-request turn has one row per distinct served model/provider
combination - the first occurrence wins, repeats are dropped, so a turn that
ping-ponged between two models shows two rows - each with its own request
count, and a blank line before the shared metrics row. These rows are
collapsed by default: the card renders only the metrics line, and clicking
the card (or pressing Pi's ctrl+o) expands it to reveal the model/provider
rows beneath. Clicking again collapses back to the single line; ctrl+o
toggles all cards at once.

Expanded (after a click or ctrl+o):

```
$0.04 | 95.5k / 380 | 99% cached

z-ai/glm-5.3 @ Modal (x3)
moonshotai/kimi-k3 @ Sail Research (x1)
```

The provider segment, the served model, and the metrics all come off the teed
response stream: the provider and the model id OpenRouter actually served off
the first chunks, and the real usage off the final chunk - the request's total
cost (rendered to plain cents; a nonzero spend never rounds down to $0.00), prompt tokens that missed provider cache / completion tokens, and the
share of the prompt served from cache. The real cost matters because pi-ai
computes cost itself from static per-model rates, which are wrong for router
slugs like `openrouter/auto` whose price varies with the routed model; the
stream's figure is preferred, with pi-ai's as fallback. The provider segment is
omitted only if the stream carried none in the brief window after the response
ends. Both lines render in the muted grey on the lightened customMessageBg
stripe, matching the per-request headers above it.

Sometimes the stream itself carries no trustworthy cost: OpenRouter bills
some requests after the fact (the generation record gets a real `total_cost`
while the stream's usage carried none — seen live with an upstream that
reported no inference cost at completion, billed $0.03 while the stream said
$0.00). The same happens when a stream lacks the provider entirely. For those
requests the card falls back to `GET /api/v1/generation?id=<gen-id>`, which
materializes a few seconds after the request. That lookup is non-blocking by
design: the session is never held up for price data — the card is simply
appended a few seconds later, once the record lands (or with best-available
data if the record never appears). No extra call is ever made when the stream
already carried complete data.

The cards are custom Pi entries: they render in the transcript and persist in
the session file, but do **not** participate in LLM context, so they never
pollute what gets sent to the model.

## Toggling the display

`/openrouter-details` flips whether the info blocks (per-response headers,
turn cards, failure blocks) are drawn in the transcript. Capture is
unaffected: every block is still captured and saved to the session file
exactly as before — only the rendering is suppressed. `/openrouter-details
on` and `/openrouter-details off` set it explicitly. The preference persists
in the session file (as an `openrouter-request-details:display` entry), so it
survives restarts, resume, and `/reload`.

## Install

```
pi install git:github.com/mitchellnemitz/pi-openrouter-request-details
```

or from a local checkout:

```
pi install /path/to/pi-openrouter-request-details
```

Reload Pi (`/reload`) or restart it to load the extension.

## Notes

- Every OpenRouter-served assistant message gets a card, whether or not the
  model was re-routed — the provider can change request-to-request even for a
  fixed model id.
- The fetch wrap is scoped to `openrouter.ai`: the metadata header is added
  only on `chat/completions` and `responses` requests, and only
  `text/event-stream` responses are teed; everything else passes through
  untouched, and the background parse is silent on every error path.
- On the `/responses` wire shape the provider is not top-level; the tee reads
  it out of the routing metadata block instead.
- Headless `pi -p` exits once the run ends, so a card that is waiting on the
  generation-record fallback may never appear in headless mode; in an
  interactive session the process stays alive and the card lands normally.
- If a future Pi-ai starts passing an explicit `fetch` into its client, the
  wrapper stops seeing requests and the card degrades to model-only. The
  permanent fix is teaching Pi-ai to capture `chunk.provider` next to
  `responseModel`.
- Requires a Pi version that exposes `responseModel`/`responseId` on the
  assistant message and the `agent_settled` extension event.
