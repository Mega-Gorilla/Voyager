FROM python:3.11-slim

# ── ① 共通ツール ＋ Node.js ───────────────────────────────────────
RUN apt-get update && apt-get install -y \
    curl gnupg build-essential git \
    python3-dev python3-pip iputils-ping

RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs

# ── ② Playwright が要求するシステムライブラリ ────────────────────
# ドキュメント: https://playwright.dev/python/docs/faq#install-deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libnss3 libnspr4 libdbus-1-3 \
        libatk1.0-0 libatk-bridge2.0-0 libatspi2.0-0 \
        libxcomposite1 libxdamage1 libxfixes3 libxrandr2 \
        libgbm1 libxkbcommon0 libasound2 \
    && rm -rf /var/lib/apt/lists/*

# ── ③ Python パッケージ ──────────────────────────────────────────
RUN pip install -U autogen-agentchat autogen-ext[openai] autogen-agentchat[lmm]~=0.2 autogen-agentchat[gemini]~=0.2

WORKDIR /app
COPY requirements.txt README.md ./
RUN pip install -r requirements.txt

# ── ④ Playwright ブラウザ (Chromium) ─────────────────────────────
# 依存ライブラリはすでに入れたので --with-deps は不要
RUN pip install playwright && playwright install chromium

WORKDIR /app/mineflayer
RUN npm install \
        mineflayer \
        mineflayer-collectblock \
        mineflayer-pathfinder \
        mineflayer-web-inventory \
        mineflayer-tool \
        mineflayer-pvp \
        prismarine-viewer \
        canvas

WORKDIR /app
