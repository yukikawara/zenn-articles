---
title: "LlamaIndex で遊んでみる"
emoji: "🦙"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["openai", "chatgpt", "llamaindex"]
published: false
---

こんにちは、始めましての方は初めまして。株式会社 Fusic の瓦です。

## 事前準備
LlamaIndex では OpenAI の API を使用します。そのために API KEY が必要となるので作成しておきましょう。作成方法は簡単で、

1. 上のプロフィールアイコンをクリック
2. `View API Keys` をクリック
3. `+ Create new secret key` をクリック

で発行できます。

※ 発行後に **API key generated** というポップアップが出てきますが、ここ以外で API KEY を見ることは出来ないので、コピーして保存するなりしてください。まあ簡単に作成、失効できるので忘れたら作り直せば済みますが…

## LlamaIndex 