---
title: "Visual ChatGPT でインタラクティブに画像を操作してみる"
emoji: "🎨"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["ChatGPT", "ComputerVision", "VisualChatGPT"]
published: true
publication_name: "fusic"
---

こんにちは、初めましての方は初めまして。株式会社 [Fusic](https://fusic.co.jp/) の瓦です。最近雨が多くて気が滅入るなと思って色々調べていたら、この季節の雨のことを「春霖」や「菜種梅雨」というらしいと知りました。

最近 ChatGPT が話題ですね。その勢いは言語を扱うだけに留まらず、三月には対話形式で画像に関する操作を行える [Visual ChatGPT](https://github.com/microsoft/visual-chatgpt) が Microsoft から公開されました。この記事では Visual ChatGPT を試してみたいと思います。

## Visual ChatGPT の簡単な説明

![](/images/try-visual-chatgpt/system_overview.png)
[https://arxiv.org/abs/2303.04671](https://arxiv.org/abs/2303.04671) の図 1 より引用

Visual ChatGPT ではまず、ユーザーの入力とこれまでの履歴をプロンプトマネージャーに渡し、プロンプトを生成します。このプロンプト用いて ChatGPT から文を出力し、画像に関する操作を行うか決定します。画像に関する操作を行った場合はまたプロンプトマネージャーからプロンプトを生成します。これを、画像に関する操作を行う必要がなくなるまで繰り返すことで、ユーザーの入力に沿った画像編集が出来るという仕組みになっているようです。

[Github](https://github.com/microsoft/visual-chatgpt) の README の [GPU memory usage](https://github.com/microsoft/visual-chatgpt#gpu-memory-usage) を見たところ、現在 (2023/04/06 時点) では以下のモデルが使用できるようです。 
| モデル | 説明 |
| ---- | ---- |
| ImageEditing | 画像編集（主に物体を消したり別の物体に入れ替えたりする） |
| InstructPix2Pix | テキストによる画像の編集（スタイルの変換など） |
| Text2Image | テキストから画像の生成 |
| ImageCaptioning | 画像からテキストの生成 |
| Image2Canny | 画像のエッジ検出 |
| CannyText2Image | エッジ検出画像とテキストから画像の生成 |
| Image2Line | 画像の直線（？）検出 |
| LineText2Image | 直線画像（？）とテキストから画像の生成 |
| Image2Hed | 画像のエッジ検出 |
| HedText2Image | エッジ検出画像とテキストから画像の生成 |
| Image2Scribble | 画像から下書き（スケッチ？）の生成 |
| ScribbleText2Image | 下書きとテキストから画像の生成 |
| Image2Pose | 画像から人体のポーズの検出 |
| PoseText2Image | ポーズ画像とテキストから画像の生成 |
| Image2Seg | 画像のセグメンテーション検出 |
| SegText2Image | セグメンテーションとテキストから画像の生成 |
| Image2Depth | 画像の深度推定 |
| DepthText2Image | 深度推定画像とテキストから画像の生成 |
| Image2Normal | 画像の法線マップの生成 |
| NormalText2Image | 法線マップとテキストから画像の生成 |
| VisualQuestionAnswering | 画像に基づいた質問応答 |

色々なモデルが用意されていることが分かります。使用する際は、`Text2Image_cuda:0` のように使用するモデルと使いたいデバイスをアンダーバーで繋いで渡すといいようです（[Quick Start](https://github.com/microsoft/visual-chatgpt#quick-start) を参照）

## Google Colab で試してみる
[README](https://github.com/microsoft/visual-chatgpt) に colab のサンプルがあるので、そちらで試してみたいと思います。使い方はとても簡単で、`%env OPENAI_API_KEY=your_key` の `your_key` を自身の OpenAI API の API KEY に変えて実行するだけです。今回読み込むモデルは

```python
"Text2Image_cuda:0,ImageCaptioning_cuda:0,ScribbleText2Image_cuda:0,Image2Scribble_cpu,Image2Canny_cpu,Image2Line_cpu,Image2Pose_cpu,Image2Depth_cpu,CannyText2Image_cuda:0,InstructPix2Pix_cuda:0,Image2Seg_cuda:0"
```
として指定しました。また、有料プランに加入していないため、GPU は T4 を使用しました。

デモでは猫を生成していたので、まず猫を生成してみます。
![](/images/try-visual-chatgpt/chat1.png)

あらかわいい。README のデモでは猫を犬に入れ替えていたので、それを試してみます。

![](/images/try-visual-chatgpt/chat2.png)
入れ替えたというよりは、新しい犬を生成しているような… しかも 3D モデルを扱うソフトウェアっぽい画面が…

とりあえず最初の猫の画像をもとに操作していきます。デモでは画像からエッジ検出を行い、その画像に基づいて別の犬を生成していましたので、まずエッジ検出をしてみます。

![](/images/try-visual-chatgpt/chat3.1.png)
![](/images/try-visual-chatgpt/chat3.2.png)
今まで生成した画像名を指定することで、これまでに生成した画像を選んで編集できます。最初の猫のエッジが検出が出来ています。次にこのエッジ検出画像に基づいて違う猫を出力させてみます。

![](/images/try-visual-chatgpt/chat4.1.png)
![](/images/try-visual-chatgpt/chat4.2.png)
指示通り灰色の猫が出力されていました！

また、エッジ画像に基づいて犬を出力させてみたものが以下になります。
![](/images/try-visual-chatgpt/gen_dog.png)
ちゃんと猫のエッジ画像に沿って犬の画像を出力しています。

ログが出力されていたので、最初の猫の画像を出力させた時、最後のエッジ画像から猫を生成した時のものを見てみます。
![](/images/try-visual-chatgpt/log1.png)
![](/images/try-visual-chatgpt/log2.png)

こちらが明示的に指定したわけではないのですが、どのようなモデルでどのようなアクションを行うかを自動で決定してます（緑字の部分）。

また、これまでの対話履歴も保存されており、ChatGPT のように対話を行うことも出来ます。「猫を生成したことがありますか？」と聞くと、はいと答えただけではなく、以下のようにこれまで生成したものを挙げられています。
![](/images/try-visual-chatgpt/chat5.png)
（上で a man が入っているのはエッジ画像から人を出力させようとしたからなのですが、うまくいきませんでした）


## まとめ
人と対話しているような文を使って画像が簡単に編集でき、正直かなり驚きました。どういう編集をするかなども人の入力から決められており、どのモデルを使用するかを考える必要がない部分がとてもいい点だと思います。また、実装によって使い方の差異があり、それらを組み合わせないといけないという部分も吸収していてとても使いやすいですね。おそらくどのモデルを使用するかは ChatGPT の出力で決めているのですが、それをうまく制御しているのはプロンプトマネージャーだと思うので、余裕があればその仕組みについても見てみたいと思います。

最後に宣伝になりますが、機械学習でビジネスの成長を加速するために、[Fusic](https://fusic.co.jp/)の機械学習チームがお手伝いたします。機械学習のPoCから運用まで、すべての場面でサポートした実績があります。もし、困っている方がいましたら、ぜひ[Fusic](https://fusic.co.jp/)にご相談ください。[お問い合わせ](https://fusic.co.jp/contact/)から気軽にご連絡いただけます。また[TwitterのDM](https://twitter.com/kawara_fusic)からでも大歓迎です！
