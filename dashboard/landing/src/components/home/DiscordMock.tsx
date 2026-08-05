import type { ReactNode } from "react";
import atlLogo from "@assets/atltransparent.png";
import { STORY_PROMPT, STORY_SPECS } from "./storyline";

const AGENT_NAME = "Alpha";

type Msg = {
  id: string;
  author: "you" | "agent";
  name: string;
  time: string;
  body: ReactNode;
};

const MESSAGES: Msg[] = [
  {
    id: "1",
    author: "you",
    name: "you",
    time: "Today at 2:11 PM",
    body: "I want to build a strategy around Berkshire Hathaway’s disclosed portfolio changes — essentially copy-trading material 13F moves.",
  },
  {
    id: "2",
    author: "agent",
    name: AGENT_NAME,
    time: "Today at 2:11 PM",
    body: "Understood. We’ll treat new 13F filings as the signal source and map significant position increases/reductions into executable trades. Two constraints to set: what counts as “material,” and the evaluation window.",
  },
  {
    id: "3",
    author: "you",
    name: "you",
    time: "Today at 2:12 PM",
    body: "Filter out small trims — only act on sizeable changes. I care about signal quality more than turnover.",
  },
  {
    id: "4",
    author: "agent",
    name: AGENT_NAME,
    time: "Today at 2:12 PM",
    body: "Agreed. I’ll threshold on relative position-size deltas so minor noise doesn’t generate trades. That keeps the book closer to Berkshire’s intentional reallocations.",
  },
  {
    id: "5",
    author: "you",
    name: "you",
    time: "Today at 2:14 PM",
    body: "Summarize this into a precise trading instruction I can reuse.",
  },
  {
    id: "6",
    author: "agent",
    name: AGENT_NAME,
    time: "Today at 2:14 PM",
    body: (
      <>
        <div>Trading instruction:</div>
        <div className="discord-prompt-block">{STORY_PROMPT}</div>
        <div className="discord-msg-followup">
          Shall I run the historical backtest with these specifications?
        </div>
      </>
    ),
  },
  {
    id: "7",
    author: "you",
    name: "you",
    time: "Today at 2:15 PM",
    body: "Yes. Proceed with the backtest.",
  },
  {
    id: "8",
    author: "agent",
    name: AGENT_NAME,
    time: "Today at 2:15 PM",
    body: "Running backtest… loading filings, aligning the trade calendar, computing equity path and risk metrics.",
  },
  {
    id: "9",
    author: "agent",
    name: AGENT_NAME,
    time: "Today at 2:16 PM",
    body: (
      <>
        <div>Backtest complete. Summary report:</div>
        <div className="discord-embed discord-embed--report">
          <div className="discord-embed-bar" />
          <div className="discord-embed-body">
            <div className="discord-embed-title">Berkshire 13F copy-trade · performance report</div>
            <div className="discord-embed-desc">
              Material 13F changes · {STORY_SPECS.timePeriod} · {STORY_SPECS.initialCapital} · {STORY_SPECS.universe} · vs DJIA / S&P 500 / Buy & Hold
            </div>
            <div className="discord-embed-grid">
              <div>
                <div className="discord-embed-label">Return</div>
                <div className="discord-embed-value text-positive">{STORY_SPECS.returnPct}</div>
              </div>
              <div>
                <div className="discord-embed-label">Sharpe</div>
                <div className="discord-embed-value">{STORY_SPECS.sharpe}</div>
              </div>
              <div>
                <div className="discord-embed-label">Max DD</div>
                <div className="discord-embed-value text-destructive">{STORY_SPECS.maxDd}</div>
              </div>
              <div>
                <div className="discord-embed-label">vs Buy &amp; Hold</div>
                <div className="discord-embed-value text-positive">{STORY_SPECS.vsBuyHold}</div>
              </div>
            </div>
            <div className="discord-embed-note">
              22 trades · avg hold 63 days · universe derived from recent Berkshire 13F holdings
            </div>
            <a href="#test" className="discord-embed-link">
              View full equity curve &amp; decision log ↓
            </a>
          </div>
        </div>
      </>
    ),
  },
];

type ChannelGroup = { label: string; channels: { name: string; active?: boolean }[] };

/** Condensed from the live Agentic Trading Discord server. */
const CHANNEL_GROUPS: ChannelGroup[] = [
  {
    label: "Start Here",
    channels: [
      { name: "welcome" },
      { name: "announcements" },
      { name: "start-here" },
    ],
  },
  {
    label: "Community",
    channels: [
      { name: "general" },
      { name: "team-formation" },
      { name: "showcase" },
      { name: "talk-to-agent", active: true },
    ],
  },
  {
    label: "Development",
    channels: [
      { name: "website-dev" },
      { name: "bug-reports" },
    ],
  },
];

/** Discord-style default avatar (blurple + logo mark) — not the ATL server icon. */
function DiscordDefaultAvatar() {
  return (
    <div className="discord-avatar discord-avatar--default" aria-hidden="true">
      <svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor">
        <path d="M20.317 4.37a19.8 19.8 0 0 0-4.885-1.515.074.074 0 0 0-.079.037c-.21.375-.444.864-.608 1.25a18.27 18.27 0 0 0-5.487 0 12.64 12.64 0 0 0-.617-1.25.077.077 0 0 0-.079-.037A19.74 19.74 0 0 0 3.677 4.37a.07.07 0 0 0-.032.027C.533 9.046-.32 13.58.099 18.057a.082.082 0 0 0 .031.057 19.9 19.9 0 0 0 5.993 3.03.078.078 0 0 0 .084-.028c.462-.63.873-1.295 1.226-1.994a.076.076 0 0 0-.041-.106 13.1 13.1 0 0 1-1.872-.892.077.077 0 0 1-.008-.128c.126-.094.252-.192.373-.292a.074.074 0 0 1 .078-.01c3.928 1.793 8.18 1.793 12.062 0a.074.074 0 0 1 .079.009c.12.098.247.198.373.293a.077.077 0 0 1-.006.127 12.3 12.3 0 0 1-1.873.892.077.077 0 0 0-.041.107c.36.698.772 1.362 1.225 1.993a.076.076 0 0 0 .084.028 19.84 19.84 0 0 0 6.002-3.03.077.077 0 0 0 .032-.054c.5-5.177-.838-9.674-3.549-13.66a.061.061 0 0 0-.031-.03zM8.02 15.33c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.956-2.419 2.157-2.419 1.21 0 2.176 1.096 2.157 2.42 0 1.333-.956 2.418-2.157 2.418zm7.975 0c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.955-2.419 2.157-2.419 1.21 0 2.176 1.096 2.157 2.42 0 1.333-.946 2.418-2.157 2.418z" />
      </svg>
    </div>
  );
}

function Avatar({ author }: { author: "you" | "agent" }) {
  if (author === "agent") {
    return <DiscordDefaultAvatar />;
  }
  return (
    <div className="discord-avatar discord-avatar--you" aria-hidden="true">
      Y
    </div>
  );
}

/** Marketing mock of the Discord channel where users talk to the agent. */
export function DiscordMock() {
  return (
    <div className="discord-mock-wrap">
      <div className="discord-mock" aria-label="Discord demo: talking to the agent">
        {/* Server rail */}
        <aside className="discord-servers" aria-hidden="true">
          <div className="discord-server-pill" />
          <div className="discord-server-icon discord-server-icon--active discord-server-icon--logo">
            <img src={atlLogo} alt="" />
          </div>
          <div className="discord-server-divider" />
          <div className="discord-server-icon">+</div>
        </aside>

        {/* Channel list — mirrors live server categories */}
        <aside className="discord-channels" aria-hidden="true">
          <div className="discord-channels-header">Agentic Trading</div>
          {CHANNEL_GROUPS.map((group) => (
            <div key={group.label}>
              <div className="discord-channel-group">{group.label}</div>
              {group.channels.map((ch) => (
                <div
                  key={ch.name}
                  className={
                    ch.active
                      ? "discord-channel-item discord-channel-item--active"
                      : "discord-channel-item"
                  }
                >
                  <span className="discord-hash">#</span> {ch.name}
                </div>
              ))}
            </div>
          ))}
        </aside>

        {/* Chat */}
        <div className="discord-main">
          <header className="discord-chat-header">
            <span className="discord-hash discord-hash--lg">#</span>
            <span className="discord-channel-name">talk-to-agent</span>
            <span className="discord-chat-header-sep" />
            <span className="discord-chat-header-topic">Talk to agents · backtest ideas</span>
          </header>

          <div className="discord-messages">
            {MESSAGES.map((msg) => (
              <div key={msg.id} className="discord-msg">
                <Avatar author={msg.author} />
                <div className="discord-msg-body">
                  <div className="discord-msg-meta">
                    <span
                      className={
                        msg.author === "agent"
                          ? "discord-username discord-username--agent"
                          : "discord-username"
                      }
                    >
                      {msg.name}
                    </span>
                    {msg.author === "agent" ? (
                      <span className="discord-bot-badge" title="Agent">
                        App
                      </span>
                    ) : null}
                    <span className="discord-timestamp">{msg.time}</span>
                  </div>
                  <div className="discord-msg-text">{msg.body}</div>
                </div>
              </div>
            ))}
          </div>

          <div className="discord-composer" aria-hidden="true">
            <div className="discord-composer-box">
              Message <span className="discord-composer-channel">#talk-to-agent</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
