---
title: ""
emoji: "🤖"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["AGI", "LLM", "CV"]
published: false
publication_name: "fusic"
---

こんにちは、初めましての方は初めまして。株式会社 Fusic の瓦です。最近朝早くに起きれず、「『春眠暁を覚えず』とはこのことか」と実感しています。

この論文では *OpenAGI: When LLM Meets Domain Experts* という論文の解説を行います。著者らによる実装は [Github](https://github.com/agiresearch/OpenAGI) にあります。

# 概要
人間はしばしば単純な能力を組み合わせて複雑なタスクをこなしています。汎用的な人工知能 (AGI) の実現にはこの能力が必要だと考えられます。

ChatGPT の流行に見られるように、近年では大規模言語モデル (LLM) の発展が盛んになっています。それに伴い、[Auto-GPT](https://github.com/Significant-Gravitas/Auto-GPT) や [BabyAGI](https://github.com/yoheinakajima/babyagi) のように自律的にタスクをこなすシステムの開発もされています。これらのシステムでは LLM をタスク設計のために使用し、出力された文から次の行動を決定しています。

この論文ではその能力に目を付け、複雑なタスクをこなすために LLM で簡単なタスクを組み合わせるシステムの提案をしています。それに加えて、タスクをこなす能力を向上させるために、**Reinforcement Learning from Task Feedback (RLTF)** という手法も提案しています。これは文字通り「解き終わったタスクの評価から強化学習を行う」手法であり、0-shot や few-shot でタスクを設計するよりも、よりよいタスクの設計が出来るようになったと報告しています。

# 提案手法


# 実験
タスクを設計する LLM として `GPT-3.5-turbo`, `LLaMA-7b`, `Flan-T5-Large` を使用しています。さらに、`zero-shot`, `few-shot`, `fine-tuning`, `RLTF` の四つの手法でタスク設計の精度がどう変化するかも実験しています。

![](/images/paper-reading-openagi/tab5.png)

それぞれの評価指標は高ければ高いほどいい性能となります。表を見ると、zero-shot よりも few-shot の方がよい結果となっており、few-shot の有効性がうかがえます。また Flan-T5-Large の結果では、単純に fine-tuning するよりも RLTF の結果の方が良くなっています。

# まとめ
（そもそもどうなれば AGI と呼べるのかという議論はされていませんが）AGI の実現に向けて、この論文ではより多くの範囲のタスクを解けるようなシステムを提案しています。論文中ではテキスト、画像を取り扱っていますが、おそらく独自のタスク（例えば音声認識でテキストにし、それを要約）に対して拡張出来ると思います。まだ「汎用」ではないですが、うまくタスクを組み合わせることでより広範囲の複雑なタスクに対応できるようになると考えると、このシステムの今後の発展に期待が高まります。

最後に宣伝になりますが、機械学習でビジネスの成長を加速するために、[Fusic](https://fusic.co.jp/)の機械学習チームがお手伝いたします。機械学習のPoCから運用まで、すべての場面でサポートした実績があります。もし、困っている方がいましたら、ぜひ[Fusic](https://fusic.co.jp/)にご相談ください。[お問い合わせ](https://fusic.co.jp/contact/)からでも気軽にご連絡いただけます。また[TwitterのDM](https://twitter.com/kawara_fusic)からでも大歓迎です！
