/**
 * Show which provider and model OpenRouter actually served for each request.
 *
 * OpenRouter returns the serving provider in a top-level `provider` field on
 * every streaming chunk, next to the `model` id. Pi-ai reads `chunk.model`
 * (stored on the assistant message as `responseModel`) but drops `chunk.provider`
 * on the floor, and no extension event carries raw chunks - so the only way to
 * see the provider without changing pi-ai is to read it off the HTTP response
 * stream ourselves.
 *
 * At load we wrap `globalThis.fetch` (pi core installs undici's fetch as the
 * global and explicitly preserves caller replacements of it; the OpenAI client
 * pi-ai builds per request binds the global unless given an explicit fetch, and
 * nothing in pi's stream path passes one). On generation calls we also set
 * `X-OpenRouter-Metadata: enabled`: the chat/completions endpoint streams
 * `provider` top-level by default, but the Responses API does not - there the
 * selected provider is only in the opt-in `openrouter_metadata` block on the
 * final chunk. For openrouter.ai streaming responses we tee the SSE body: one
 * branch back to pi-ai, untouched; the other
 * parsed in the background for the generation `id`, the serving `provider`,
 * the `model` id OpenRouter actually served, and the real `usage`/cost from
 * the final chunk - all remembered in a map keyed by generation id. At
 * message_end pi gives us that same id as `responseId` (pi-ai captures
 * `chunk.id`), so rendering is a map lookup.
 *
 * A turn can contain many requests (tool-call loops, retries): every
 * assistant message is its own billed request. The card is appended once per
 * turn, at agent_settled, and aggregates the whole turn - summed usage/cost,
 * the request count, and one row per distinct served model/provider
 * combination (first occurrence wins; repeats are dropped).
 *
 * Each response also gets a one-line header ABOVE its block in the
 * transcript: the served model and provider, in place before any of the
 * streamed response renders. A placeholder entry is appended at
 * message_start - extension handlers run before the UI creates the
 * streaming component, so the entry lands directly above it - and its
 * component re-reads the tee's live state on every frame, filling in as the
 * first chunks arrive. At message_end a complete copy of the header is
 * appended (this one persists); the placeholder's data stays empty forever,
 * so a rebuilt session (reload, resume) renders only the complete copy.
 * Both the header and the aggregate card render on the customMessageBg
 * stripe (the background pi uses for its own informational notices) with
 * zero padding, so they read as system meta - not response text, and not a
 * prompt submission box.
 *
 * Pi-ai computes cost itself from static per-model rates and drops OpenRouter's
 * streamed `usage.cost`; for router slugs (openrouter/auto) whose price varies
 * with the routed model those rates are wrong, so the card prefers the real
 * streamed cost. Sometimes the stream itself has no cost: OpenRouter bills
 * some requests after the fact (the generation record gets a real total_cost
 * while the stream's usage carried none - seen live with an upstream that
 * reported no inference cost at completion). For those, and for streams that
 * lacked the provider entirely, the card falls back to
 * `GET /api/v1/generation?id=<gen-id>`, which materializes a few seconds after
 * the request. That lookup is non-blocking by design: agent_settled only
 * awaits data that is already local (the tee finishes parsing milliseconds
 * after the stream ends), and when a record lookup is needed it runs detached
 * and the card is appended when it completes - the session stays usable the
 * whole time. Nothing is ever fetched when the stream already carried
 * complete data.
 *
 * Cards are custom Pi entries: they render in the transcript and persist in
 * the session file, but do NOT participate in LLM context, so they never
 * pollute what gets sent to the model.
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import type { Component } from "@earendil-works/pi-tui";
import { Box, Container, MouseRegion, Spacer, Text, visibleWidth, wrapTextWithAnsi } from "@earendil-works/pi-tui";

const CUSTOM_TYPE = "openrouter-request-details";
// Per-response header entry: the model/provider that served THAT response,
// rendered above the response block. Appended at message_end - before pi
// persists the message - so it lands above the block both live (the TUI
// splices custom entries in front of the streaming component) and in the
// session file (entry order: header, then message).
const SERVED_TYPE = "openrouter-request-details:served";
const FAILED_TYPE = "openrouter-request-details:failed";
// Display preference. Appended whenever /openrouter-details flips visibility:
// the data entries above are always captured and persisted regardless, this
// only records whether the renderers should draw them. Read back at
// session_start so the preference survives restarts (and /reload, via the
// shared state below).
const DISPLAY_TYPE = "openrouter-request-details:display";

// One per-response header: the model and provider that served THAT response.
// `served` is optional because a placeholder entry (appended at
// message_start, before the response streams) carries empty data - its
// renderer fills live - and only the complete copy appended at message_end
// persists with values.
interface ServedHeaderData {
    served?: string;
    provider?: string;
    // Generation tag on live placeholders: matches `headerGen` of the request
    // currently streaming. A placeholder only renders while its own request
    // is live - without this, every placeholder component reads the same
    // currentHeader and all of them illuminate during the NEXT request's
    // streaming (a stale duplicate header stacked above the previous one).
    gen?: number;
}

interface RequestUsage {
    input: number;
    output: number;
    cacheRead: number;
    cost: number;
}

// One distinct served-model/provider combination within a turn, with the
// number of requests in the turn that it served.
interface ServingCombo {
    requested: string;
    served: string;
    provider?: string;
    count: number;
}

// Persisted card payload. The top-level requested/served/provider fields are
// the first combo's values, kept so entries written by older versions of this
// extension (and older entries read by newer renderers) still render.
interface RequestDetailsData {
    requested: string;
    served: string;
    provider?: string;
    usage?: RequestUsage;
    requests?: number;
    combos?: ServingCombo[];
}

// What the tee pulls out of one request's stream.
interface StreamInfo {
    provider?: string;
    model?: string;
    usage?: RequestUsage;
}

// Wrapper + parsed-info map must survive extension reloads (/reload
// re-imports the module): keyed on Symbol.for so a fresh module instance
// reuses the installed wrapper and the same map instead of stacking a second
// wrapper that feeds a map the new instance never reads. The rates catalog
// (below) rides the same state so /reload keeps it warm.
// One failed generation request: OpenRouter rejected it before routing, so it
// never appears in generations/logs - capture the exact payload + error body
// here, since this is the only place the failing request is observable.
export interface FailedRequest {
    timestamp: number;
    url: string;
    status: number;
    errorBody?: string;
    requestBody?: string;
    // Present when fetch itself rejected (network/connection error): no HTTP
    // response exists, so status is 0 and this carries the thrown error.
    fetchError?: string;
}

interface TeeState {
    wrapped: boolean;
    wrapperVersion?: number;
    originalFetch?: typeof fetch;
    providers: Map<string, StreamInfo>;
    rates?: Map<string, ModelRates>;
    rateFetches?: Map<string, Promise<void>>;
    rateAttempts?: Map<string, number>;
    /** Bumped on module reload; stale-generation fetch completions discard. */
    rateGeneration?: number;
    errors?: FailedRequest[];
    reportedErrorCount?: number;
    visible?: boolean;
}

// Bump when wrapFetch's behavior changes: /reload re-imports the module, and
// the version mismatch here replaces the previously installed wrapper instead
// of silently keeping the old one.
const WRAPPER_VERSION = 2;
const STATE_KEY = Symbol.for("openrouter-request-details");
const globalAny = globalThis as any;
if (!globalAny[STATE_KEY]) {
    globalAny[STATE_KEY] = { wrapped: false, providers: new Map() };
}
const state: TeeState = globalAny[STATE_KEY];
// Clear on module reload so stale rates from previous sessions/days don't
// persist; attempt timestamps reset with them, or a pre-reload fetch attempt's
// 60s failure cooldown would suppress the post-reload refetch and leave
// headers without prices.
state.rates = new Map();
state.rateFetches = new Map();
state.rateAttempts = new Map();
// Bump on reload: in-flight fetches started by the previous module capture
// the old generation and discard their results, so a completion from before
// the reload can neither repopulate state.rates nor delete the new module's
// rateFetches entry (which would suppress revalidation).
state.rateGeneration = (state.rateGeneration ?? 0) + 1;
state.errors ??= [];
state.visible ??= true;

// Bound the map: gen ids are unique per request, so long sessions would
// otherwise grow it without end.
const MAX_REMEMBERED = 200;

function rememberInfo(genId: string, patch: Partial<StreamInfo>) {
    const isNew = !state.providers.has(genId);
    state.providers.set(genId, { ...state.providers.get(genId), ...patch });
    if (isNew && state.providers.size > MAX_REMEMBERED) {
        const oldest = state.providers.keys().next().value;
        if (oldest !== undefined) state.providers.delete(oldest);
    }
}

// Drain the tee branch and pull what the card needs out of the SSE lines:
// generation id and serving provider (first chunks), the model id OpenRouter
// actually served, and the real usage from the final chunk. Handles both
// OpenRouter stream shapes: chat completions puts id/model/provider/usage at
// the top level of each chunk; the Responses API nests them under `response`
// (and never sends `provider` at all). Silent by design: this branch is
// diagnostics only and must never be able to affect the response pi-ai sees.
async function consumeSSE(body: ReadableStream<Uint8Array>) {
    let genId: string | null = null;
    let provider: string | null = null;
    let model: string | null = null;
    let usage: RequestUsage | null = null;
    try {
        const reader = body.getReader();
        const decoder = new TextDecoder();
        let buf = "";
        for (;;) {
            const { done, value } = await reader.read();
            if (done) break;
            buf += decoder.decode(value, { stream: true });
            let nl = buf.indexOf("\n");
            while (nl >= 0) {
                const line = buf.slice(0, nl).replace(/\r$/, "");
                buf = buf.slice(nl + 1);
                nl = buf.indexOf("\n");
                if (!line.startsWith("data:")) continue;
                const payload = line.slice(5).trim();
                if (!payload || payload === "[DONE]") continue;
                if (genId && provider && model && usage) continue; // have what we need; just drain
                let chunk: any;
                try {
                    chunk = JSON.parse(payload);
                } catch {
                    continue;
                }
                const info = chunk.response ?? chunk;
                const u = info.usage;
                const patch: Partial<StreamInfo> = {};
                if (!genId && typeof info.id === "string" && info.id) genId = info.id;
                if (!provider && typeof chunk.provider === "string" && chunk.provider) {
                    provider = chunk.provider;
                    patch.provider = chunk.provider;
                }
                // The opt-in metadata block (header injected in wrapFetch)
                // carries the selected provider when the stream shape omits it:
                // top-level on completions chunks, nested in `response` on the
                // Responses API.
                if (
                    !provider &&
                    Array.isArray(info.openrouter_metadata?.endpoints?.available)
                ) {
                    const selected = info.openrouter_metadata.endpoints.available.find(
                        (e: any) => e?.selected === true,
                    );
                    if (typeof selected?.provider === "string" && selected.provider) {
                        provider = selected.provider;
                        patch.provider = selected.provider;
                    }
                }
                if (!model && typeof info.model === "string" && info.model) {
                    model = info.model;
                    patch.model = info.model;
                }
                if (
                    !usage &&
                    u &&
                    typeof u === "object" &&
                    (typeof u.prompt_tokens === "number" || typeof u.input_tokens === "number")
                ) {
                    const prompt = u.prompt_tokens ?? u.input_tokens ?? 0;
                    const cached =
                        u.prompt_tokens_details?.cached_tokens ??
                        u.input_tokens_details?.cached_tokens ??
                        0;
                    usage = {
                        input: Math.max(0, prompt - cached),
                        output: u.completion_tokens ?? u.output_tokens ?? 0,
                        cacheRead: cached,
                        cost: typeof u.cost === "number" ? u.cost : 0,
                    };
                    patch.usage = usage;
                }
                if (genId && Object.keys(patch).length > 0) rememberInfo(genId, patch);
            }
        }
    } catch {
        // ignore - see above
    }
}

// True for streaming calls to OpenRouter, whatever the endpoint: chat
// completions, the Responses API, or anything new OpenRouter adds. Non-SSE
// responses (model lists, generation records) are never teed - the wrap only
// branches on event-stream responses - so this check is cheap insurance, not
// a gate we have to maintain per endpoint.
function isOpenRouterChatURL(input: string | URL | Request): boolean {
    try {
        const raw =
            typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
        return new URL(raw).host === "openrouter.ai";
    } catch {
        return false;
    }
}

// The generation endpoints that stream `provider` data worth having. Other
// openrouter.ai URLs (model lists, generation records) are left untouched.
function isGenerationEndpoint(input: string | URL | Request): boolean {
    try {
        const raw =
            typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
        const path = new URL(raw).pathname;
        return path.endsWith("/chat/completions") || path.endsWith("/responses");
    } catch {
        return false;
    }
}

function wrapFetch(originalFetch: typeof fetch): typeof fetch {
    return (input, init) => {
        try {
        if (isOpenRouterChatURL(input) && isGenerationEndpoint(input)) {
            // Opt into routing metadata: on the Responses API this is the only
            // place the serving provider appears. Harmless on completions,
            // where `provider` is already top-level on every chunk.
            const headers = new Headers(init?.headers);
            headers.set("X-OpenRouter-Metadata", "enabled");
            init = { ...init, headers };
        }
        return originalFetch(input, init).then((response) => {
                if (!isOpenRouterChatURL(input)) return response;
                if (!response.ok) {
                    // Generation request rejected before routing (e.g. Responses
                    // API validation 400): nothing reaches generations/logs, so
                    // clone and capture the exact error + request payload here.
                    if (isGenerationEndpoint(input)) {
                        void response
                            .clone()
                            .text()
                            .then((errorBody) => {
                                const rawBody = typeof init?.body === "string" ? init.body : undefined;
                                state.errors ??= [];
                                state.errors.push({
                                    timestamp: Date.now(),
                                    url: String(typeof input === "string" ? input : input instanceof URL ? input.href : input.url),
                                    status: response.status,
                                    errorBody: errorBody.slice(0, 2000),
                                    requestBody: rawBody?.slice(0, 4000),
                                });
                                if (state.errors.length > 10) state.errors.shift();
                            })
                            .catch(() => {});
                    }
                    return response;
                }
                if (!response.body) return response;
                const contentType = response.headers.get("content-type") ?? "";
                if (!contentType.includes("text/event-stream")) return response;
                    const [piBranch, teeBranch] = response.body.tee();
                    void consumeSSE(teeBranch);
                    return new Response(piBranch, {
                        status: response.status,
                        statusText: response.statusText,
                        headers: response.headers,
                });
            });
        } catch (err: any) {
            // fetch rejected (connection error, DNS, TLS): no response object
            // exists, so nothing else will record it. Capture and rethrow.
            if (isOpenRouterChatURL(input) && isGenerationEndpoint(input)) {
                const rawBody = typeof init?.body === "string" ? init.body : undefined;
                state.errors ??= [];
                state.errors.push({
                    timestamp: Date.now(),
                    url: String(typeof input === "string" ? input : input instanceof URL ? input.href : input.url),
                    status: 0,
                    fetchError: String(err?.message ?? err),
                    requestBody: rawBody?.slice(0, 4000),
                });
                if (state.errors.length > 10) state.errors.shift();
            }
            throw err;
        }
    };
}

if (!state.wrapped || state.wrapperVersion !== WRAPPER_VERSION) {
    // First install: capture the true original. Version upgrade: the currently
    // installed (older-version) wrapper becomes the delegate — it still fills
    // the shared state, so one stale layer is harmless and no further stacking
    // occurs on later reloads.
    state.originalFetch ??= globalThis.fetch;
    globalThis.fetch = wrapFetch(state.originalFetch);
    state.wrapped = true;
    state.wrapperVersion = WRAPPER_VERSION;
}

// The tee branch usually finishes parsing within milliseconds of pi-ai
// consuming its branch, but agent_settled can beat it; poll briefly instead of
// rendering a card with no provider. This is local data, never a network wait.
// A miss degrades to whatever the tee did catch, or pi-ai's data alone.
const PROVIDER_WAIT_MS = 1500;
const PROVIDER_POLL_MS = 50;

async function lookupInfo(genId: string): Promise<StreamInfo | undefined> {
    const deadline = Date.now() + PROVIDER_WAIT_MS;
    for (;;) {
        const hit = state.providers.get(genId);
        if (hit && hit.provider && hit.usage) return hit;
        if (Date.now() >= deadline) return hit;
        await new Promise((r) => setTimeout(r, PROVIDER_POLL_MS));
    }
}

// The per-response header needs only the model and provider, which ride the
// FIRST chunks of the stream - long parsed by the time message_end fires.
// A short bounded poll covers the rare race where the tee branch has not
// been scheduled yet; unlike lookupInfo it never waits for usage, so the
// tool loop is not stalled behind the final chunk.
const SERVED_WAIT_MS = 250;
const SERVED_POLL_MS = 25;

async function lookupServed(genId: string): Promise<StreamInfo | undefined> {
    const deadline = Date.now() + SERVED_WAIT_MS;
    for (;;) {
        const hit = state.providers.get(genId);
        if (hit && hit.provider && hit.model) return hit;
        if (Date.now() >= deadline) return hit;
        await new Promise((r) => setTimeout(r, SERVED_POLL_MS));
    }
}

// --- Per-response header -----------------------------------------------------
//
// The header above each streamed response. `currentHeader` is the live state
// for the request currently streaming: the tee's first chunks land in it
// within milliseconds, and the placeholder's component re-reads it on every
// frame, so the line is in place while the response text is still streaming
// in below it.

let currentHeader: ServedHeaderData | null = null;
// Bumped at every message_start; a placeholder renders only while
// currentHeader belongs to its own request.
let headerGen = 0;

// Data objects of placeholders appended by THIS module instance. Only these
// get the live component: on a rebuilt session (reload, resume) the renderer
// sees entry data from a previous module instance or from the session file,
// neither of which is in this set, so stale placeholders render nothing and
// only the complete message_end copies show.
const livePlaceholders = new WeakSet<object>();

// One info-block row: "served @ provider(suffix)" in a single fg wrap, so
// the whole line is one color. Every info block renders in the same muted
// grey, clearly dimmer than response text.
function infoRow(
    served: string,
    provider: string | undefined,
    suffix: string,
    theme: any,
    color: string,
): string {
    return theme.fg(color, `${served}${provider ? ` @ ${provider}` : ""}${suffix}`);
}

// Both info blocks render on a lightened customMessageBg stripe - the
// background pi uses for its own informational notices (compaction
// summaries, skill invocations), lifted 20% so it stands out against the
// terminal background without turning into a highlight. The Box adds one cell
// of padding on every side, so the text never touches the stripe edge.
function infoBgFn(theme: any): (text: string) => string {
    // theme.bg wraps text in the theme's ANSI sequence; a probe with empty
    // text yields the raw background escape to lighten. Falls back to the
    // theme token when the sequence is not a truecolor rgb.
    const probe = theme.bg("customMessageBg", "");
    const m = probe.match(/^\x1b\[48;2;(\d+);(\d+);(\d+)m/);
    if (!m) return (text: string) => theme.bg("customMessageBg", text);
    const lift = (c: number) => Math.min(255, Math.round(c * 1.2));
    const rgb = `\x1b[48;2;${lift(+m[1])};${lift(+m[2])};${lift(+m[3])}m`;
    return (text: string) => `${rgb}${text}\x1b[49m`;
}

// Right-aligned text component: wraps lines and pads with leading spaces so
// the content sits against the right margin of the full-width Box stripe.
class RightAlignedText implements Component {
    constructor(private text: string = "") {}

    setText(text: string): void {
        this.text = text;
    }

    invalidate(): void {}

    render(width: number): string[] {
        if (!this.text || this.text.trim() === "") return [];
        const lines = wrapTextWithAnsi(this.text, width);
        return lines.map((line) => {
            const len = visibleWidth(line);
            const pad = Math.max(0, width - len);
            return " ".repeat(pad) + line;
        });
    }
}

// A pi-tui component whose render() reads currentHeader on every frame.
// Containers re-render their children unconditionally each frame (Box only
// caches the final lines), so no invalidation plumbing is needed: the moment
// the tee's data lands in currentHeader, the line appears. Rendering goes
// through an inner RightAlignedText so the live line gets the exact same
// background fill and right-alignment as the persisted copy.
class LiveHeaderText implements Component {
    private box = new Box(1, 0);
    private text = new RightAlignedText("");
    private last = "";

    constructor(private theme: any, private gen: number) {
        this.box.setBgFn(infoBgFn(theme));
        this.box.addChild(this.text);
    }

    render(width: number): string[] {
        const h = currentHeader;
        const line =
            h && h.gen === this.gen && h.served
                ? infoRow(h.served, h.provider, ratesSuffix(h.served, h.provider), this.theme, "muted")
                : "";
        if (line !== this.last) {
            this.last = line;
            this.text.setText(line);
        }
        return this.box.render(width);
    }

    invalidate(): void {
        this.text.invalidate();
    }
}

// --- Provider rates catalog -------------------------------------------------
//
// The "before" block shows the serving provider's listed rates. Nothing we
// already receive carries them: the stream metadata has only provider/model/
// selected, and the model-level /models pricing is the cheapest endpoint's
// rate. The only per-provider source is the endpoints catalog,
// /api/v1/models/{author}/{slug}/endpoints - public, ~40KB per model, with
// prompt/completion/cache-read per-token prices per provider. Fetched once at
// session start for the startup model and whenever a response is served by a
// provider the cached catalog does not list (cooldown-limited), so a normal
// session costs one fetch per distinct served model.

interface ProviderRates {
    prompt: number; // USD per token
    completion: number;
    cacheRead: number;
}

interface ModelRates {
    // provider -> distinct priced variants (a provider can list several
    // quantization tiers at different rates; duplicate rows are exact dupes)
    rates: Map<string, ProviderRates[]>;
    fetchedAt: number;
}

// Catalog exists but lacks the served provider: refetch sparingly. No catalog
// at all (startup, or the last fetch failed): retry sooner.
// Pricing changes on OpenRouter (discounts, tier shifts) - expire catalog rates
// after 15 minutes so headers don't display immortalized stale numbers.
const RATE_TTL_MS = 15 * 60 * 1000;
const RATE_REFETCH_COOLDOWN_MS = 10 * 60 * 1000;
const RATE_FAILURE_COOLDOWN_MS = 60 * 1000;

// Router aliases have no endpoints catalog; concrete author/slug ids do.
function catalogModelId(model: string): string | null {
    let id = model;
    while (id.startsWith("openrouter/")) id = id.slice("openrouter/".length);
    if (!id || id === "auto" || id === "auto-beta" || !id.includes("/")) return null;
    return id;
}

function fetchModelRates(id: string): Promise<void> {
    const gen = state.rateGeneration;
    const existing = state.rateFetches!.get(id);
    if (existing) return existing;
    state.rateAttempts!.set(id, Date.now());
    const attempt = (async () => {
        try {
            const key = process.env.OPENROUTER_API_KEY;
            const res = await fetch(`https://openrouter.ai/api/v1/models/${id}/endpoints`, {
                headers: key ? { Authorization: `Bearer ${key}` } : undefined,
            });
            if (!res.ok) throw new Error(String(res.status));
            const json: any = await res.json();
            const endpoints = json?.data?.endpoints;
            if (!Array.isArray(endpoints)) throw new Error("unexpected shape");
            const rates = new Map<string, ProviderRates[]>();
            for (const e of endpoints) {
                const name = e?.provider_name;
                const p = e?.pricing;
                if (typeof name !== "string" || !p) continue;
                const r: ProviderRates = {
                    prompt: Number(p.prompt),
                    completion: Number(p.completion),
                    cacheRead: Number(p.input_cache_read ?? 0),
                };
                if (!Number.isFinite(r.prompt) || !Number.isFinite(r.completion)) continue;
                const list = rates.get(name) ?? [];
                if (
                    !list.some(
                        (x) =>
                            x.prompt === r.prompt &&
                            x.completion === r.completion &&
                            x.cacheRead === r.cacheRead,
                    )
                ) {
                    list.push(r);
                    rates.set(name, list);
                }
            }
            // Discard pre-reload completions: the fresh catalog revalidation
            // will refetch.
            if (gen === state.rateGeneration) state.rates!.set(id, { rates, fetchedAt: Date.now() });
        } catch {
            // Leave uncached; the attempt timestamp rate-limits the next try.
        } finally {
            if (gen === state.rateGeneration) state.rateFetches!.delete(id);
        }
    })();
    state.rateFetches!.set(id, attempt);
    return attempt;
}

// Make sure the catalog for `served` is on hand. Detached by contract: the
// caller may fire-and-forget, or await the returned promise bounded (see
// ratesReady). A provider change within a cached model costs nothing - the
// catalog already lists every provider; only a provider the catalog predates
// triggers a refetch.
function ensureRates(served: string, provider?: string): Promise<void> | undefined {
    const id = catalogModelId(served);
    if (!id) return undefined;
    const inflight = state.rateFetches!.get(id);
    if (inflight) return inflight;
    const cat = state.rates!.get(id);
    const isExpired = cat ? Date.now() - cat.fetchedAt > RATE_TTL_MS : true;
    if (cat && !isExpired && (!provider || cat.rates.has(provider))) return undefined;
    const cooldown = cat && !isExpired ? RATE_REFETCH_COOLDOWN_MS : RATE_FAILURE_COOLDOWN_MS;
    if (Date.now() - (state.rateAttempts!.get(id) ?? 0) < cooldown) return undefined;
    return fetchModelRates(id);
}

// Rates fragment for the header line: "($0.91 in / $2.86 out / $0.17 cache)".
// Empty when the catalog has nothing for this provider yet (fetch pending,
// failed, or provider unknown) or the cached entry is past the TTL — an
// expired entry renders nothing rather than stale prices; the background
// revalidation repopulates the suffix on the next rebuild. A provider with
// several priced quantization variants renders a min-max range.
function fmtRate(v: number): string {
    const perM = v * 1_000_000;
    return `$${perM === 0 || perM >= 0.005 ? perM.toFixed(2) : perM.toFixed(4)}`;
}

function ratesSuffix(served: string | undefined, provider: string | undefined): string {
    if (!served || !provider) return "";
    const id = catalogModelId(served);
    const cat = id ? state.rates!.get(id) : undefined;
    if (!cat || Date.now() - cat.fetchedAt > RATE_TTL_MS) return "";
    const variants = cat.rates.get(provider) ?? [];
    if (variants.length === 0) return "";
    const seg = (pick: (r: ProviderRates) => number) => {
        const vals = [...new Set(variants.map(pick))].sort((a, b) => a - b);
        if (vals.length === 1) return fmtRate(vals[0]);
        return `${fmtRate(vals[0])}-${fmtRate(vals[vals.length - 1]).slice(1)}`;
    };
    return ` (${seg((r) => r.prompt)} / ${seg((r) => r.completion)} / ${seg(
        (r) => r.cacheRead,
    )})`;
}

// Bounded wait for an in-flight catalog fetch, so a header persisted right
// after a cold fetch still carries rates. Gives up fast - 250ms, the same cap
// as the served lookup - because a missing suffix only costs that one entry.
async function ratesReady(served: string): Promise<void> {
    const id = catalogModelId(served);
    const inflight = id ? state.rateFetches!.get(id) : undefined;
    if (!inflight) return;
    await Promise.race([inflight, new Promise((r) => setTimeout(r, 250))]);
}

// --- Turn aggregation -------------------------------------------------------
//
// A turn (one user message through agent_settled) can contain many OpenRouter
// requests: each assistant message in a tool-call loop is a separate billed
// request. Collect every request at message_end, then build one card for the
// whole turn at agent_settled.

interface TurnRequest {
    requested: string;
    responseModel?: string;
    responseId?: string;
    usage?: RequestUsage; // pi-ai's computed accounting; the stream's is preferred
}

let turnRequests: TurnRequest[] = [];

// One request with all sources merged: stream info (preferred) over pi-ai's
// message data. `suspect` marks requests whose data the stream could not be
// trusted for (no usage, or a zero cost alongside real tokens, or no provider)
// and that therefore need the generation-record lookup.
interface MergedRequest {
    requested: string;
    responseId?: string;
    served: string;
    provider?: string;
    usage?: RequestUsage;
    suspect: boolean;
}

function mergeTurn(requests: TurnRequest[], infos: (StreamInfo | undefined)[]): MergedRequest[] {
    return requests.map((r, i) => {
        const info = infos[i];
        const usage = info?.usage ?? r.usage;
        const provider = info?.provider;
        const tokens = usage ? usage.input + usage.cacheRead + usage.output : 0;
        const suspect =
            !!r.responseId &&
            (!usage || (usage.cost <= 0 && tokens > 0) || !provider);
        return {
            requested: r.requested,
            responseId: r.responseId,
            served: info?.model || r.responseModel || r.requested,
            ...(provider ? { provider } : {}),
            ...(usage ? { usage } : {}),
            suspect,
        };
    });
}

function buildCardData(items: MergedRequest[]): RequestDetailsData {
    const total: RequestUsage = { input: 0, output: 0, cacheRead: 0, cost: 0 };
    let anyUsage = false;
    const counts = new Map<string, number>();
    for (const item of items) {
        const key = `${item.requested}\u0000${item.served}\u0000${item.provider ?? ""}`;
        counts.set(key, (counts.get(key) ?? 0) + 1);
    }
    const combos: ServingCombo[] = [];
    const seen = new Set<string>();
    for (const item of items) {
        if (item.usage) {
            anyUsage = true;
            total.input += item.usage.input;
            total.output += item.usage.output;
            total.cacheRead += item.usage.cacheRead;
            total.cost += item.usage.cost;
        }
        const key = `${item.requested}\u0000${item.served}\u0000${item.provider ?? ""}`;
        if (seen.has(key)) continue;
        seen.add(key);
        combos.push({
            requested: item.requested,
            served: item.served,
            ...(item.provider ? { provider: item.provider } : {}),
            count: counts.get(key) ?? 1,
        });
    }
    return {
        requested: items[0].requested,
        served: combos[0].served,
        ...(combos[0].provider ? { provider: combos[0].provider } : {}),
        ...(anyUsage ? { usage: total } : {}),
        requests: items.length,
        combos,
    };
}

// --- Generation-record fallback ----------------------------------------------

interface GenerationRecord {
    total_cost?: number;
    provider_name?: string;
    native_tokens_prompt?: number | null;
    native_tokens_completion?: number | null;
    native_tokens_cached?: number | null;
}

// The record lags the request: it 404s for roughly the first two seconds and
// is reliably present by about eight. Poll with a bounded backoff, then give
// up - the card is appended with whatever is known rather than waiting
// forever. This runs detached from agent_settled; the session is never held
// up for it.
const RECORD_POLL_DELAYS_MS = [2000, 2000, 3000, 3000, 4000];

async function fetchGenerationRecord(genId: string): Promise<GenerationRecord | null> {
    const key = process.env.OPENROUTER_API_KEY;
    if (!key) return null;
    for (const delay of [0, ...RECORD_POLL_DELAYS_MS]) {
        if (delay) await new Promise((r) => setTimeout(r, delay));
        try {
            const res = await fetch(
                `https://openrouter.ai/api/v1/generation?id=${encodeURIComponent(genId)}`,
                { headers: { Authorization: `Bearer ${key}` } },
            );
            if (res.status === 404) continue; // record not written yet
            if (!res.ok) return null; // auth/quirk failures: stop, retrying won't help
            const json: any = await res.json();
            const data = json?.data;
            return data && typeof data === "object" ? (data as GenerationRecord) : null;
        } catch {
            // network hiccup - fall through to the next delay
        }
    }
    return null;
}

// Fill suspect requests from their generation records. Tokens in the record
// are native (provider-tokenizer) counts, which is what OpenRouter streams in
// usage too; cost comes from total_cost, the billed figure.
async function correctSuspects(items: MergedRequest[]): Promise<MergedRequest[]> {
    const out = items.map((item) => ({ ...item }));
    const suspects = out.filter((item) => item.suspect && item.responseId);
    const records = await Promise.all(
        suspects.map((item) => fetchGenerationRecord(item.responseId!)),
    );
    suspects.forEach((item, i) => {
        const rec = records[i];
        if (!rec) return;
        if (typeof rec.total_cost === "number") {
            if (!item.usage) {
                const prompt = rec.native_tokens_prompt ?? 0;
                const cached = rec.native_tokens_cached ?? 0;
                item.usage = {
                    input: Math.max(0, prompt - cached),
                    output: rec.native_tokens_completion ?? 0,
                    cacheRead: cached,
                    cost: rec.total_cost,
                };
            } else {
                item.usage = { ...item.usage, cost: rec.total_cost };
            }
        }
        if (!item.provider && typeof rec.provider_name === "string" && rec.provider_name) {
            item.provider = rec.provider_name;
        }
    });
    return out;
}

// --- Rendering ----------------------------------------------------------------

function fmtTokens(n: number): string {
    if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
    if (n >= 10_000) return `${Math.round(n / 1_000)}k`;
    if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
    return String(n);
}

// One row per distinct served model/provider combination, each with its
// request count. A turn that touched several models reads badly as a single
// wrapped line: each combination gets its own row instead, before the shared
// metrics row.
function modelLines(data: RequestDetailsData, theme: any): RightAlignedText[] {
    const combos: ServingCombo[] = data.combos ?? [
        {
            requested: data.requested,
            served: data.served,
            ...(data.provider ? { provider: data.provider } : {}),
            count: data.requests ?? 1,
        },
    ];
    return combos.map(
        (combo) =>
            new RightAlignedText(
                infoRow(
                    combo.served,
                    combo.provider,
                    ` (x${combo.count})`,
                    theme,
                    "muted",
                ),
            ),
    );
}

function usageLine(data: RequestDetailsData, theme: any): RightAlignedText | null {
    if (!data.usage) return null;
    const u = data.usage;
    const promptTokens = u.input + u.cacheRead;
    const cachePct = promptTokens > 0 ? Math.round((u.cacheRead / promptTokens) * 100) : 0;
    return new RightAlignedText(
        theme.fg(
            "muted",
            // Cost to plain cents; a nonzero spend never renders as $0.00.
            `$${u.cost > 0 && u.cost < 0.005 ? "0.01" : u.cost.toFixed(2)} | ${fmtTokens(u.input)} / ${fmtTokens(u.output)} | ${cachePct}% cached`,
        ),
    );
}

// -- Extension wiring ----------------------------------------------------------

export default function (pi: ExtensionAPI) {
    pi.registerCommand("openrouter-details", {
        description: "Toggle OpenRouter request-details info blocks (capture always continues)",
        getArgumentCompletions: (prefix: string) => {
            const envs = ["on", "off"].filter((v) => v.startsWith(prefix));
            return envs.length > 0
                ? envs.map((v) => ({ value: v, label: v }))
                : null;
        },
        handler: async (args, ctx) => {
            const arg = args?.trim().toLowerCase();
            if (arg === "on" || arg === "off") {
                state.visible = arg === "on";
            } else if (arg) {
                ctx.ui.notify("usage: /openrouter-details [on|off]", "error");
                return;
            } else {
                state.visible = !state.visible;
            }
            try {
                pi.appendEntry(DISPLAY_TYPE, { visible: state.visible });
            } catch {
                // stale context - the toggle still applies to this session
            }
            // Pi invokes each custom entry's renderer once, when its transcript
            // component is built, so flipping the flag alone only affects
            // entries rendered later. Rebinding the same theme instance rides
            // pi's theme-change path (ui.invalidate()), which cascades into
            // every mounted component and re-runs all entry renderers under
            // the new flag - the same rebuild pi's own ctrl+t toggle gets.
            ctx.ui.setTheme(ctx.ui.theme);
            ctx.ui.notify(
                `OpenRouter request details ${state.visible ? "shown" : "hidden"} (capture continues)`,
                "info",
            );
        },
    });

class ClickableSummaryComponent extends Container {
    private expanded: boolean;
    constructor(
        private data: RequestDetailsData,
        private theme: any,
        initialExpanded: boolean,
        private onToggle?: (expanded: boolean) => void,
    ) {
        super();
        this.expanded = initialExpanded;
        this.renderContent();
    }

    private renderContent() {
        this.clear();
        const box = new Box(1, 0, infoBgFn(this.theme));
        const usage = usageLine(this.data, this.theme);
        if (usage) {
            box.addChild(usage);
        }
        if (this.expanded) {
            if (usage) box.addChild(new Spacer(1));
            for (const line of modelLines(this.data, this.theme)) box.addChild(line);
        } else if (!usage) {
            for (const line of modelLines(this.data, this.theme)) box.addChild(line);
        }

        this.addChild(
            new MouseRegion(box, (event) => {
                if (event.type === "click" && event.button === "left") {
                    this.expanded = !this.expanded;
                    this.renderContent();
                    this.onToggle?.(this.expanded);
                    return { handled: true };
                }
                return undefined;
            }),
        );
    }
}

    const clickOverrides = new WeakMap<object, boolean>();
    const entryGlobalExpanded = new WeakMap<object, boolean>();

    pi.registerEntryRenderer(CUSTOM_TYPE, (entry, options, theme) => {
        if (!state.visible) return undefined;
        const data = entry.data as RequestDetailsData;

        // When ctrl+o changes the global expanded state for this entry,
        // clear the individual click override so the global action takes precedence.
        const lastGlobal = entryGlobalExpanded.get(entry);
        if (lastGlobal !== undefined && lastGlobal !== options.expanded) {
            clickOverrides.delete(entry);
        }
        entryGlobalExpanded.set(entry, options.expanded);

        const initialExpanded = clickOverrides.get(entry) ?? options.expanded;

        return new ClickableSummaryComponent(data, theme, initialExpanded, (expanded) => {
            clickOverrides.set(entry, expanded);
        });
    });

    pi.registerEntryRenderer(FAILED_TYPE, (entry, _opts, theme) => {
        if (!state.visible) return undefined;
        const data = entry.data as FailedRequest | undefined;
        if (!data) return undefined;
        const box = new Box(1, 0, infoBgFn(theme));
        if (data.fetchError) {
            box.addChild(new Text(`OpenRouter request FAILED (connection): ${data.fetchError.slice(0, 300)}`, 0, 0));
        } else {
            box.addChild(new Text(`OpenRouter request FAILED (${data.status})`, 0, 0));
            box.addChild(new Text(`error: ${data.errorBody?.slice(0, 300) ?? "unknown"}`, 0, 0));
        }
        if (data.requestBody) {
            box.addChild(new Text(`request: ${data.requestBody.slice(0, 600)}`, 0, 0));
        }
        return box;
    });

    pi.registerEntryRenderer(SERVED_TYPE, (entry, _opts, theme) => {
        if (!state.visible) return undefined;
        const data = entry.data as ServedHeaderData | undefined;
        // Complete header (appended at message_end): render from data, with
        // the provider's listed rates when the catalog has them.
        if (data?.served) {
            const box = new Box(1, 0, infoBgFn(theme));
            box.addChild(
                new RightAlignedText(
                    infoRow(
                        data.served,
                        data.provider,
                        ratesSuffix(data.served, data.provider),
                        theme,
                        "muted",
                    ),
                ),
            );
            return box;
        }
        // Live placeholder for the response currently streaming: a
        // component that re-reads currentHeader every frame. Anything else -
        // an empty placeholder on a rebuilt session - renders nothing.
        if (data && livePlaceholders.has(data)) {
            return new LiveHeaderText(theme, data.gen ?? 0);
        }
        return undefined;
    });

    pi.on("message_start", async (event) => {
        const m = event.message;
        if (m.role !== "assistant") return;
        if (m.provider !== "openrouter") return;
        // Placeholder above the about-to-stream response. Extension handlers
        // run before the UI adds the streaming component, so the entry lands
        // directly above it; the live component fills it from the tee as the
        // first chunks arrive. Data stays empty - the complete copy appended
        // at message_end is the one that persists.
        currentHeader = { gen: ++headerGen };
        const placeholder: ServedHeaderData = { gen: headerGen };
        livePlaceholders.add(placeholder);
        pi.appendEntry(SERVED_TYPE, placeholder);
    });

    pi.on("message_update", async (event) => {
        const m = event.message;
        if (m.role !== "assistant") return;
        if (m.provider !== "openrouter") return;
        const h = currentHeader;
        if (!h || h.served || !m.responseId) return;
        // Fill the live header as soon as the tee has the first chunks.
        // Detached: never hold up the stream for display data.
        void lookupServed(m.responseId).then((info) => {
            if (currentHeader !== h) return; // stream ended or new request
            h.served = info?.model || m.responseModel || m.model;
            if (info?.provider) h.provider = info.provider;
            // Warm the rates catalog for whatever actually served this.
            void ensureRates(h.served, h.provider);
        });
    });

    pi.on("message_end", async (event) => {
        const m = event.message;
        if (m.role !== "assistant") return;
        if (m.provider !== "openrouter") return;
        // Aborted attempts persist with zeroed usage and no responseId; skip
        // them rather than letting noise inflate the turn count.
        if (!m.responseId) {
            currentHeader = null;
            return;
        }

        turnRequests.push({
            requested: m.model,
            responseModel: m.responseModel || undefined,
            responseId: m.responseId,
            usage: m.usage
                ? {
                      input: m.usage.input,
                      output: m.usage.output,
                      cacheRead: m.usage.cacheRead,
                      cost: m.usage.cost.total,
                  }
                : undefined,
        });

        // The complete, persisted header. The live state is usually filled
        // long before the stream ends; the bounded lookup only runs in the
        // rare case it is not (a stream so short the fill never landed).
        const h = currentHeader;
        let served = h?.served;
        let provider = h?.provider;
        if (!served || !provider) {
            const info = await lookupServed(m.responseId);
            served = served || info?.model || m.responseModel || m.model;
            provider = provider || info?.provider;
        }
        currentHeader = null; // blanks the live placeholder
        // Persist with rates when they are seconds away (cold catalog fetch
        // kicked by this very response); never wait longer than a blink.
        void ensureRates(served, provider);
        await ratesReady(served);
        pi.appendEntry(SERVED_TYPE, {
            served,
            ...(provider ? { provider } : {}),
        });
    });

    pi.on("session_start", async (_event: unknown, ctx: any) => {
        turnRequests = [];
        currentHeader = null;
        // Restore the display preference: last display entry in the file wins.
        for (const entry of ctx?.sessionManager?.getEntries?.() ?? []) {
            if (
                entry.type === "custom" &&
                entry.customType === DISPLAY_TYPE &&
                typeof (entry.data as any)?.visible === "boolean"
            ) {
                state.visible = (entry.data as any).visible;
            }
        }
        // Warm the rates catalog for the startup model so the first header
        // already carries pricing.
        const model = ctx?.getModel?.();
        const id = model ? catalogModelId(String((model as any).id ?? model)) : null;
        if (id) void ensureRates(id);
    });

    pi.on("session_shutdown", async () => {
        turnRequests = [];
        currentHeader = null;
    });

    pi.on("agent_settled", async () => {
        const failures = (state.errors ?? []).slice(state.reportedErrorCount ?? 0);
        state.reportedErrorCount = (state.errors ?? []).length;
        for (const failure of failures) {
            try {
                pi.appendEntry(FAILED_TYPE, failure);
            } catch {
                // stale context - the entry still lands next turn
            }
        }
        const requests = turnRequests;
        turnRequests = [];
        if (requests.length === 0) return;
        const infos = await Promise.all(
            requests.map((r) => (r.responseId ? lookupInfo(r.responseId) : Promise.resolve(undefined))),
        );
        const merged = mergeTurn(requests, infos);
        if (merged.every((item) => !item.suspect)) {
            appendCard(pi, buildCardData(merged));
            return;
        }
        // Some requests need the generation-record fallback. Run it detached:
        // the session is usable immediately, and the card appends when the
        // records land (or with best-available data once the poll gives up).
        void correctSuspects(merged)
            .then((fixed) => appendCard(pi, buildCardData(fixed)))
            .catch(() => appendCard(pi, buildCardData(merged)));
    });
}

// Appending is safe outside event handlers (pi only rejects calls on a stale
// context after session replacement or /reload); still guard, so a delayed
// card never throws into an already-settled turn.
function appendCard(pi: ExtensionAPI, data: RequestDetailsData) {
    try {
        pi.appendEntry(CUSTOM_TYPE, data);
    } catch {
        // stale context (session replaced / reloaded mid-lookup) - drop the card
    }
}
