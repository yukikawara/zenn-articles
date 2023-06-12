---
title: "色々な大規模言語モデルを試してみる"
emoji: "💻"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["LLM", "NLP", "ML"]
published: true
publication_name: fusic
---

こんにちは、初めましての方は初めまして。株式会社 Fusic の瓦です。

ChatGPT が提供されてからというもの、世間の言語モデルに対する注目は留まるところを知らず、様々な団体や企業による言語モデルの利用や開発のニュースを見ない日は一日もないくらいです。この記事ではそのような企業で開発されているモデルを動かしてみたものになります。

:::message
この記事では、公開されているデータセットを用いた定量的な評価はしていません。そのような結果が気になる方は、Stability AI 社が [github で公開](https://github.com/Stability-AI/lm-evaluation-harness/tree/jp-stable)していますのでそちらを確認していただければと思います。
:::


## 環境

試してみたモデルは以下の六種類になります。

- [cyberagent/open-calm-1b](https://huggingface.co/cyberagent/open-calm-1b): CyberAgent 社が開発したモデル
- [cyberagent/open-calm-3b](https://huggingface.co/cyberagent/open-calm-3b): CyberAgent 社が開発したモデル
- [cyberagent/open-calm-7b](https://huggingface.co/cyberagent/open-calm-7b): CyberAgent 社が開発したモデル
- [rinna/japanese-gpt-neox-3.6b](https://huggingface.co/rinna/japanese-gpt-neox-3.6b): Rinna 社が開発したモデル
- [rinna/japanese-gpt-neox-3.6b-instruction-sft-v2](https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-sft-v2): Rinna 社が開発したモデル
- [rinna/japanese-gpt-neox-3.6b-instruction-ppo](https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-ppo): Rinna 社が開発したモデル

また CALM を動かしたのは AWS の `g4dn.2xlarge` で、rinna のモデルを動かしたのは `g4dn.4xlarge` です。rinna のモデルは `g4dn.2xlarge` でもギリギリ GPU には乗るのですが、推論させると Out of memory が起きたので大きいインスタンスで動かしています。AMI は pytorch 1.13.1 が入った ubuntu ベースの深層学習用のもの (`ami-078e2151eadb5c30c`) を使いました。

## 実際に試してみる

まず環境構築から始めます。といっても必要なものはほとんど入っているので、EC2 を起動した後に

```bash
source activate pytorch
python -m pip install huggingface accelerate sentencepiece
```

で必要なものが揃います。`accelerate` は CALM に、`sentencepiece` は rinna のモデルに必要なので、片方のモデルだけ使いたい場合はそれぞれ必要なものだけで充分です。

必要なライブラリは揃ったので、実際にモデルを動かします。動かし方が少しずつ違うので、CALM 用、rinna の普通の 3.6b 用、rinna の instruction tuning したモデル用のコードをそれぞれ用意します。特に rinna の instruction tuning したものは会話形式で入力するべきであり、注意点として

1. 登場するのは「ユーザー」と「システム」である。
2. 会話の発言は「{ユーザー, システム}: {発言}」という形で表される。
3. それぞれの発言を `<NL>` で繋げたものがモデルへの入力となる。

が挙げられています。より詳しくは [Rinna の Huggingface](https://huggingface.co/rinna) のモデルのページをご参照ください。

```python:CALM 用コード
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "cyberagent/open-calm-1b"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, device_map="auto", torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

text = "こんにちは、"

inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    tokens = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.pad_token_id,
    )

output = tokenizer.decode(tokens[0], skip_special_tokens=True)
print(output)
```

```python:rinna-3.6b 用
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "rinna/japanese-gpt-neox-3.6b"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

text = "こんにちは、"

inputs = tokenizer(
    text, add_special_tokens=False, return_tensors="pt"
).to(model.device)

with torch.no_grad():
    tokens = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

output = tokenizer.decode(tokens[0])
print(output)
```

```python:rinna-3.6b-instruction 用
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "rinna/japanese-gpt-neox-3.6b-instruction-sft-v2"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

text = "ユーザー: こんにちは"
text = f"{text}<NL>システム: "

inputs = tokenizer(
    text, add_special_tokens=False, return_tensors="pt"
).to(model.device)

with torch.no_grad():
    tokens = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

output = tokenizer.decode(tokens[0])
print(output)
```

`temperature` や `repetition_penalty` は各モデルのページの値に基づいています。ここら辺を変えることでより良いと思う文を生成させることも出来ると思いますが、今回はこの部分の試行錯誤はせずに文を生成させてみます。

それでは実際に文の生成をさせていきます。今回は「物語の生成」「料理のレシピの生成」「適切な返答の生成」をさせてみました。またそれぞれのモデルを指す略称として
- cyberagent/open-calm-1b: CALM-1B
- cyberagent/open-calm-3b: CALM-3B
- cyberagent/open-calm-7b: CALM-7B
- rinna/japanese-gpt-neox-3.6b: Rinna-3.6B
- rinna/japanese-gpt-neox-3.6b-instruction-sft-v2: Rinna-SFT
- rinna/japanese-gpt-neox-3.6b-instruction-ppo: Rinna-PPO

を使用します。

`generate` 関数の引数に `do_sample=True` を指定しているので実行のたびに生成結果が異なることに注意です。何回も試すことでより適切な文を生成出来ることもあると思いますが、今回は生成は三回行い、主観で選んだ良さそうな一文を載せています。

### 物語の生成
「昔々あるところに、」という文を与え、どのような続きを生成するか見てみます。`Rinna-SFT` と `Rinna-PPO` には「ユーザー: 「昔々あるところに」から始まる物語を考えてください」という入力を与えました。

| モデル | 出力 |
| ---- | ---- |
| CALM-1B | 昔々あるところに、小さな村がありました。そこに1人の男の子が住んでいました。「お兄ちゃん」と慕っていたその「兄貴」、このときはまだ名前もなく、「オーキン」「オクチン」...呼び名すらありませんでした...。しかしその子は立派な青年に成長していました! ある日突然家に帰る途中に転んでしまったのです...!! |
| CALM-3B | 昔々あるところに、1つの国がありました。その国は3人の王様で成り立っていましたが、「悪」という悪魔が王様を狙い襲来し「善(平民)」である国民たちを苦しめていました。「お前には罪があるんだ!」と責め続けましたが誰も聞く耳を持ちません......。そしてついに最後の悪魔がやって きて・・・『魔女の宅急便』とは角野栄子による児童書で、『第75回毎日出版文化賞』『平成28年度青少年読書感想文全国コンクール高等学校部門課題図書』、『2016年ブッカー国際賞最優秀新人賞受賞作』、さらに実写映画化されたことで大きな話題となった作品です。『キキの夢には何があった?』で はそんな作品の世界を垣間見る事ができますよ! |
| CALM-7B | 昔々あるところに、と始まり、歴史小説や史実を脚色した時代劇ではよく使われる。物語の語り手が過去を振り返る時の定番ともいえる。「俺も昔は......」で話は終わるとすれば、それは悲劇になる可能性が高い。「あれからずっと後悔している」と言い切るのは辛いものだ。しかし、そう言い切らざるを得ないほど、今の人生はつらいものなのだろう。主人公の心中を思うと胸が痛む。だからこそ、最後は明るい結末になってほしい。 |
| Rinna-3.6B | 昔々あるところに、魔女がいました」とか「魔法使いのおばあさんが魔法の杖をふりました」とか、それなりに「○○の魔法使い」という存在は昔話や童話で見かけるので、そういった存在がいるということは知っている方も多いのではないでしょうか。 その魔法使いは、「どんなことだってで きちゃう」という魔法を使えるのですが、ある時その「どんなことだってできちゃう」魔法を使うことをやめてしまいます。 そして、今のように 「魔法」は特別な力や儀式なんかで使うものではなく、みんなが普通に使えるものだったのです。 それはなぜかというと、魔法使いは、自分一人 で魔法を使うことをやめて、自分の魔法がどんなものなのかということを、自分の周りにいるたくさんの人に教えるようになったのです。 そのた くさんの人の中に、あなたが入ってくれるというのなら、私があなたの願いをその魔法の力で叶えることができます。 そう、あなたが魔法使いに 言って、魔法使いが願いを魔法で叶えてくれるのです。 |
| Rinna-SFT | ユーザー: 「昔々あるところに」から始まる物語を考えてください。<NL>システム: もちろんです。以下はいくつかの例です:<NL><NL>「昔々、私は森の大きな木に住んでいました。」<NL><NL>「昔々、私は小さな家に一人で住んでいました。」<NL><NL>「昔々、私は森の中で暮らしていました。」<NL><NL>「昔々、私は大きな家に住んでいました。」 |
| Rinna-PPO | ユーザー: 「昔々あるところに」から始まる物語を考えてください。<NL>システム: 「昔々あるところに、森があった」という文は正しいです。森には多くの木々があり、川や海もあります。これらのものはすべて、地球が形成された後に初めて生まれたものです。その後、長い時間が経過し、地球は現在の姿になりました。最終的な答えは「はい」となります。 |

内容としては微妙なものもありますが、どのモデルでも流暢な文が生成されています。`Rinna-PPO` ではなぜか答えを出力していますが何を聞かれたんだろう…

### レシピの生成
「最近暑いので、トマトとバジルを使ったさっぱりしたレシピを教えます。まず」という文を与え、どのような続きを生成するか見てみます。`Rinna-SFT` と `Rinna-PPO` には「ユーザー: 最近暑いので、トマトとバジルを使ったさっぱりしたレシピを教えてください。」という入力を与えました。

| モデル | 出力 |
| ---- | ---- |
| CALM-1B | 最近暑いので、トマトとバジルを使ったさっぱりしたレシピを教えます。まずナスを輪切りにして水にさらしザルに上げ水気を切ります(中まで火が通らないため)次にフライパンでオリーブオイルを熱してニンニクのみじん切りにんにくを入れて香りが出たらみょうが・しょうがも千切って入 れてさっと炒めますそして赤唐辛子の輪を切りとって入れますその上から塩コショウをして最後にケチャップを入れ全体に絡めれば出来上がりです!!お好みで一味や七味もかけても美味しいですよおはようございます!今日のオススメは『パプリカ』を使って作るお料理をご紹介します♪作り方はとっても簡単!『油揚げ&ミニにんたまチーズ炒め』『ツナとマヨネーズ和え』、『鶏胸肉の甘辛煮』、他にも色々ありますよ~★☆ |
| CALM-3B | 最近暑いので、トマトとバジルを使ったさっぱりしたレシピを教えます。まず鶏むね肉を一口大に切り(フォークで刺しても良い)、皮目に格子状に切れ目を入れてから調味料を塗り込んで10分ほどおきます。(片面焼き) 次に玉ねぎをみじん切りにします。。フライパンにオリーブオイルを入れ中火で熱し玉葱を投入して炒めますが焦げ付きやすいので木べらなどでゆっくり混ぜながら焼きます。*今回は1個ずつフライパンに入れてます! |
| CALM-7B | 最近暑いので、トマトとバジルを使ったさっぱりしたレシピを教えます。まず玉ねぎをスライスして水にさらします。水気を切った玉ねぎとツナ缶とマヨネーズを混ぜて塩コショウで味を調えて冷蔵庫に入れておきます。お皿に半分に折ったレタスを敷いて、真ん中にトマトを盛り付けます。もう1枚のレタスをお皿のまわりに添えたら出来上がりです。ちょっと酸味のある生ハムやチーズと一緒に食べても美味しいですし、オリーブオイルをかけてもおいしいです。お好みでお酢をかけてもいいと思います。 |
| Rinna-3.6B | 最近暑いので、トマトとバジルを使ったさっぱりしたレシピを教えます。まずトマトは湯剥きして、種を除いて5mmくらいの角切りにします。次にバジルをみじん切りにして、オリーブオイルと塩、コショウ、酢で調味してください。このままでも食べられますが、少しオリーブオイルを足して、バジルの風味を強めてもいいですね。これを器に入れて冷蔵庫で冷やしましょう。 暑い夏にさっぱりとした食事をしたいときに、おすすめです 。 |
| Rinna-SFT | ユーザー: 最近暑いので、トマトとバジルを使ったさっぱりしたレシピを教えてください。<NL>システム: もちろんです!以下の材料を用意してみましょう。<NL><NL>• スパゲッティ・ボロネーゼ(ミートソース)<NL>• トマトとモッツァレラチーズのスパゲッティ<NL>• ズッキーニとアスパラガスのグリル<NL>• アボカドとツナのサラダ<NL>• グリーンビーンズとライス<NL>• フレンチフライドポテト |
| Rinna-PPO | ユーザー: 最近暑いので、トマトとバジルを使ったさっぱりしたレシピを教えてください。<NL>システム: もちろんです。トマトとバジルを混ぜ合わせて冷製パスタにしたり、冷やしてサラダにかけたりすることができます。また、レモン汁と砂糖を加えて、ヨーグルトに混ぜ込んでマリネ液を作ることもできます。塩とコショウで味付けし、冷蔵庫で冷やしてから提供します。 |

`CALM-1B` や `CALM-3B` では「トマトとバジルは何処へ…？」というレシピが生成されていますが、こちらも流暢な日本語が生成されています。
`Rinna-SFT` では上と同じように色んな選択肢を出力しています。fine tuning 用のデータにそういうものが多いんですかね？

### 返答の生成
「「さっきの映画めっちゃ泣いちゃった～」という言葉に対して、適切な返事は」という文を与え、どのような続きを生成するか見てみます。`Rinna-SFT` と `Rinna-PPO` には「ユーザー: 「さっきの映画めっちゃ泣いちゃった～」という言葉に対する適切な返事を教えてください。」という入力を与えました。

| モデル | 出力 |
| ---- | ---- |
| CALM-1B | 「さっきの映画めっちゃ泣いちゃった~」という言葉に対して、適切な返事はできません。でもそんなときこそ、「ちょっと待って」と一言言って相手の話を聞く余裕がほしいですね。「どうしたの?」「何があったの?」って聞くだけじゃなくて相手が話しやすい雰囲気を作ってあげるのも優しさですよね! |
| CALM-3B | 「さっきの映画めっちゃ泣いちゃった~」という言葉に対して、適切な返事は?【悲報】ジャンプ『サムライ8』打ち切りの可能性キタ―――(゚∀゚)――――!! |
| CALM-7B | 「さっきの映画めっちゃ泣いちゃった~」という言葉に対して、適切な返事はどのようなものなのでしょうか? ちょっと考えてみましょう。まず、「泣く」というのは感情の伝達なので、言葉で伝えるのは難しいですよね。そこで、代わりに「涙を拭く」「ハンカチを差し出す」といった行動で返すのが良いでしょう。「ティッシュください!」という一言も、伝えやすいですね。ただ、これらはあくまでも無難な言葉です。特に男性は、女性以上に傷つきやすいです。男性の心を落ち着かせるためにも、自分の感情をしっかり伝えてあげるようにしましょう。そして、相手はそれをしっかり受け止めてあげてください。きっと彼も、あなたのやさしさに感謝してくれるはずです。 |
| Rinna-3.6B | 「さっきの映画めっちゃ泣いちゃった～」という言葉に対して、適切な返事はなかなか出てきません...。 「あそこ泣けたよね～」「僕も泣いちゃった」という言い方はできても、 それは、相手が話している「泣く」という感覚の部分を「感情」と捉えることができていないからです。 そういうときには、泣いた原因や泣いたことに対しての気持ちを聞いてあげましょう。 「何で泣いちゃったんだろうね?」とその気持ちを聞いてあげることができます。 この2つの方法を身に着け、様々な場面で活用できるようになると、 |
| Rinna-SFT | ユーザー: 「さっきの映画めっちゃ泣いちゃった～」という言葉に対する適切な返事を教えてください。<NL>システム: 私は、その言葉を言われたときにどのような気持ちになるかによって異なりますが、「素晴らしい映画だった!」と言ってほしいかもしれません。 |
| Rinna-PPO | ユーザー: 「さっきの映画めっちゃ泣いちゃった～」という言葉に対する適切な返事を教えてください。<NL>システム: ありがとうございます!その言葉はとても素晴らしく、感動的な映画を指しているようですね。良い映画を観る機会に感謝します。素晴らしい一日を過ごしてくださいね。良い一日を! |

個人的には `Rinna-SFT` の返答が一番欲しいものに近いです。上のようにいくつか選択肢を出してくれると嬉しいんですが、今回試したときはそのような文は生成できませんでした。

## まとめ
この記事では大規模言語モデルを動かし、生成される文を見てみました。`CALM-1B` のモデルでも思っていたより流暢な日本語が生成されており、fine tuning することで生成させたい文に近づけることも出来そうな気がします。`Rinna-PPO` は性能がよいらしいですが、fine tuning をして生成する文を調整することが出来るのかは今後試してみたいところです。

最後に宣伝になりますが、機械学習でビジネスの成長を加速するために、[Fusic](https://fusic.co.jp/)の機械学習チームがお手伝いたします。機械学習のPoCから運用まで、すべての場面でサポートした実績があります。もし、困っている方がいましたら、ぜひ[Fusic](https://fusic.co.jp/)にご相談ください。[お問い合わせ](https://fusic.co.jp/contact/)からでも気軽にご連絡いただけます。また[TwitterのDM](https://twitter.com/kawara_fusic)からでも大歓迎です！
