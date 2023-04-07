---
title: "LoRA: Low-Rank Adaptaion of Large Language Models の解説"
emoji: "📑"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["NLP", "LLM"]
published: false
publication_name: "fusic"
---

こんにちは、初めましての方は初めまして。株式会社 [Fusic](https://fusic.co.jp/) の瓦です。


この記事では [LoRA: Low-Rank Adaptaion of Large Language Models](https://arxiv.org/abs/2106.09685) (以降 LoRA として参照) の解説をします。忙しい方は[概要](#概要)だけ読んでください。

# 概要
一言で言えば、LoRA は**効率的な追加学習手法の一つ**です。単純にファインチューニングを行うと、訓練時にモデルのパラメータを全て保持しつつパラメータの更新を行わなければならないため、ベースにするモデルによっては莫大なメモリが必要となります。また、追加で学習させたいタスクそれぞれに対してモデルが必要となり、デプロイ時には学習させたタスクの分だけサーバを用意しなければなりません。特に大きなモデルを使用するとその分強いサーバを用意しなければならず、莫大なコストがかかってしまいます。

![](/images/paper-reading-lora/overview.png)
Fig.1 より引用

この問題に対して、論文では元のパラメータを更新せずに、差分を計算するモデルを学習するアプローチを提案しています。上図のように、元のパラメータに対して、行列分解するモデルを学習させています。この手法によって、GPT-3 を単純に追加学習する場合に比べて学習に必要なパラメータ数は 1/10000 になり、使用する GPU のメモリは 1/3 になっているとのことです。また、他の先行研究と比較して学習に必要なパラメータが少ないにもかかわらず、匹敵する結果、またはより良い結果を出しています。

RoBERTa や DeBERTa、GPT-2 の学習済みモデルや実装は [Github のページ](https://github.com/microsoft/LoRA) に載っています。

# 提案手法


# 実験結果


# まとめ

最後に宣伝になりますが、機械学習でビジネスの成長を加速するために、[Fusic](https://fusic.co.jp/)の機械学習チームがお手伝いたします。機械学習のPoCから運用まで、すべての場面でサポートした実績があります。もし、困っている方がいましたら、ぜひ[Fusic](https://fusic.co.jp/)にご相談ください。[お問い合わせ](https://fusic.co.jp/contact/)からでも気軽にご連絡いただけます。また[TwitterのDM](https://twitter.com/kawara_fusic)からでも大歓迎です！

