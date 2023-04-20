---
title: "Inf2 インスタンスが GA になったので触ってみる"
emoji: "💻"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["AWS", "Cloud", "Inf2", "LLM"]
published: true
publication_name: "fusic"
---

こんにちは、初めましての方は初めまして。株式会社 Fusic の瓦です。最近 AWS の Inf2 の一般提供が開始されたり、AWS Bedrock の発表があったり、

# Inf2 について

Inf2 インスタンスは**推論に特化したインスタンス**です。元々推論用のインスタンスとしては Inf1 インスタンスがありました。しかし、最近のモデルは急速な大規模化が進んでおり、Inf1 インスタンスではホストできないような大きなモデルも増えてきています。そのためなのか、Inf1 インスタンスと比較して **Inf2 インスタンスではメモリ量が大幅に増加**しています（最大で 384 GB）[^1]。また、スループットも多くなっており、より速い推論が可能となっています。このことによって、**コストは大幅に削減できつつスループットは大幅に増加**できているらしいです[^2]。

[^1]: [Inf2 インスタンスのページ](https://aws.amazon.com/jp/ec2/instance-types/inf2/)
[^2]: [Announcing New Tools for Building with Generative AI on AWS](https://aws.amazon.com/jp/blogs/machine-learning/announcing-new-tools-for-building-with-generative-ai-on-aws/)

# 試してみる

「[大規模モデル推論コンテナを使って AWS Inferentia2 に大規模言語モデルをデプロイ](https://aws.amazon.com/jp/blogs/news/deploy-large-language-models-on-aws-inferentia2-using-large-model-inference-containers/)」を参考にして大規模言語モデルのデプロイを試してみます。この記事では `OPT-13B` を `inf2.48xlarge` インスタンスにデプロイしているのですが、モデルのサイズを見てみると `inf2.8xlarge` でも動きそうなので、そちらで試してみます。

## Inf2 インスタンスの起動

記事の「Inferentia ハードウェアを起動」の通りに起動します。ここで注意が必要で、Inf2 インスタンスは今のところオハイオリージョン、バージニア北部リージョンでしか利用できません。また、4/20 時点ではバージニア北部リージョンでしか inf インスタンスの上限緩和の申請が出来ませんでした。なので、より大きなインスタンスを使用したい場合はバージニア北部リージョンで利用するといいと思います。

## コードの配置

記事の「必要な依存関係のインストールとモデルの作成」の通りに動かします。記事では Jupyter Notebook を使用していますが、面倒だったので CLI で操作しました。記事の通りにディレクトリ作成やファイルの作成を行うと、以下のような構成になると思います。

```text
.
├── logs
└── models
    └── opt13b
        ├── serving.properties
        └── model.py
```

`serving.properties` は記事と同じ設定にします（その他の設定については「[すべての DJL 設定オプション](https://github.com/deepjavalibrary/djl-serving/blob/master/serving/docs/configurations.md)」を参照すればいいとのことです）`models.py` はそのままだと、バッチ内のテキストの長さが異なるため Tensor に出来ずにエラーが起きます。そのため、`infer` 関数の `input_ids = ...`, `outputs = ...` の部分を

```python
input_ids = tokenizer.batch_encode_plus(prompt, padding=True, return_tensors="pt").input_ids
...
outputs = tokenizer.batch_decode(generated_sequence, skip_special_tokens=True)
```

にします。長い文に合わせてパディングされるので短い文には余計なトークンが入ってしまいますが、これでエラーは出なくなります。

## サービング用のコンテナ起動

記事の「サービングコンテナを実行」の通りに実行します。`ls /dev/ | grep neuron` を実行すると `inf2.8xlarge` ではデバイスが一つしかないことが分かるので、`docker run` 時の `--device` 引数は対応する一つだけを渡せば大丈夫です。

コンテナでは、モデルを Huggingface からダウンロードしてコンパイルし、メモリに載せることで推論可能な状態にします。モデルのダウンロード、およびコンパイルはモデルのサイズによって変化します。`OPT-13B` ではかなり時間がかかりました（25 分ほど）コンパイル中は出力が止まったりターミナルの反応が遅くなったりするのですが、気長に待ちましょう。

コンパイルまで終わると、8080 番ポートを叩いて使えるようになります。記事では推論用の API を叩いていますが、サービングで使用されている DJL では他の API も多数用意されています。[DJL のリポジトリの REST API](https://github.com/deepjavalibrary/djl-serving#rest-api) に他の API の説明が載っているので気になる方はそちらを参照してください。

## 実際に叩いてみる

まずは記事のサンプルを叩いてみました。

```bash
curl -X POST "http://127.0.0.1:8080/predictions/opt13b" \
     -H 'Content-Type: application/json' \
     -d '{"seq_length":2048,
          "text":[
                    "Hello, I am a language model,",
                    "Welcome to Amazon Elastic Compute Cloud,"
                 ]
         }'
```

`seq_length` を 2048 に設定しているせいか、かなりの長文が返ってきます。また、`time` コマンドで時間を計測してみたところ、2m36.323s かかっており、かなり時間がかかっていることが分かります。

OPT では一応多言語も対応しているので、日本語の推論をさせてみます。投げたテキストは

```text
日本語から英語へ翻訳してください。
こんにちは -> Hello
パンが食べたい。 -> I want to eat bread.
LLM はその生成能力からテキスト合成、要約、機械翻訳などのタスクに広く用いられています。 -> LLM is widely used for tasks such as text synthesis, summarization, and machine translation due to its generative capabilities.
本記事で扱う範囲について、以下の点に注意してください。 ->
```

で、few-shot による日英翻訳をさせてみようとしたものになっています。返ってきた結果は

```text
日本語から英語へ翻訳してください。
こんにちは -> Hello
パンが食べたい。 -> I want to eat bread.
LLM はその生成能力からテキスト合成、要約、機械翻訳などのタスクに広く用いられています。 -> LLM is widely used for tasks such as text synthesis, summarization, and machine translation due to its generative capabilities.
本記事で扱う範囲について、以下の点に注意してください。 -> 【本記事についてはここで位置づける】
LLMを用いても、実用的なテキスト作成との互換性は非常に点数で済む言語互換ですし、例えば「茨城に行く」という言葉にLLMを使用すれば、「行く」という表現に「行」のマスを変えたり、「茨城あたり」は「茨城さき」にしたりといった出来事が可能となります。また、謎の「有機」を連想させているからであれば、LLMを使用すれば英語入門の問題を抱えてしまう可能性もあります。 -> 【LLMを用いても、
```

この先も文が続くのですが、長いのでカットしました。そもそも日本語の文章がおかしく、日本語を上手く扱えないようです。

# まとめ

この記事では Inf2 インスタンスを触り、実際に LLM のデプロイまでを行ってみました。最低の `inf2.xlarge` でもメモリが 32 GB あり、`inf2.48xlarge` ではメモリが 384 GB あるなど、最近の大規模なモデルにより特化したインスタンスという感じです。推論用のチップに載せるためにモデルのコンパイルを行わなければならず、そのために対応出来るモデルが限られている点はあるものの、かなり簡単に LLM をデプロイできる環境が手に入るという点ではとても素晴らしいと思います。コンパイルにかなり時間がかかるのがネックだと感じたのですが、恐らく先にコンパイルして S3 に置いておいて、利用時に S3 からダウンロードするなどすれば一回一回コンパイルが終わるのを待つ必要はないのかなと思います。これについては今後調査してみたいと思います。

最後に宣伝になりますが、機械学習でビジネスの成長を加速するために、[Fusic](https://fusic.co.jp/)の機械学習チームがお手伝いたします。機械学習のPoCから運用まで、すべての場面でサポートした実績があります。もし、困っている方がいましたら、ぜひ[Fusic](https://fusic.co.jp/)にご相談ください。[お問い合わせ](https://fusic.co.jp/contact/)からでも気軽にご連絡いただけます。また[TwitterのDM](https://twitter.com/kawara_fusic)からでも大歓迎です！
