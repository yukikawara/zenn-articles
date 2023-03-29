---
title: "LlamaIndex で遊んでみる"
emoji: "🦙"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["OpenAI", "ChatGPT", "LlamaIndex"]
published: true
---

こんにちは、初めましての方は初めまして。株式会社 Fusic の瓦です。春ですね。

2022 年 12 月に ChatGPT が突如現れてから、大規模言語モデル (LLM) を使ったアプリやライブラリがたくさん出てきました。ChatGPT はとても流暢な文を生成出来るのですが、訓練に含まれていない情報（専門的な知識や学習後に出た情報）について生成出来ないという弱点があります。その弱点を克服するものの一つとして、**LlamaIndex** というライブラリが公開されています。この記事では、その LlamaIndex を試してみたいと思います。

## 事前準備
LlamaIndex で ChatGPT を使用するためには、OpenAI の API KEY が必要です。そのため、まずは OpenAI のアカウントを作って API KEY を作成しておきましょう。作成方法は簡単で、ログイン後に

1. 上のプロフィールアイコンをクリック
2. `View API Keys` をクリック
3. `+ Create new secret key` をクリック

で発行できます。

※ 発行後に **API key generated** というポップアップが出てきますが、ここ以外で API KEY を見ることは出来ないので、コピーして保存するなりしてください。まあ簡単に作成、失効できるので忘れたら作り直せば済みますが…

## LlamaIndex を試してみる
LlamaIndex は、**自分の持っているデータや専門知識などの外部データを LLM に簡単に組み込めるライブラリ**です。似たようなライブラリとしては [LangChain](https://github.com/hwchase17/langchain) がありますが、LlamaIndex の方がより簡単に触れそうなのでこちらを使用します。

LlamaIndex には、指定したディレクトリ以下のファイルをデータとして読み込むクラス `SimpleDirectoryReader` が提供されています。以下のコードで、データの読み込みおよびインデックスが作成できます。
```python
from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader
from llama_index.llm_predictor.chatgpt import ChatGPTLLMPredictor

documents =  SimpleDirectoryReader("/path/to/directory").load_data()
index = GPTSimpleVectorIndex(documents=documents, llm_predictor=ChatGPTLLMPredictor())
```

`index` 作成は Open AI の Embeddins の API に投げて結果を取得しています。そのため、データ量に応じて時間がかかります。 

これで準備は終了です。とても簡単ですね。**インデックスは毎回作成していると時間とお金がかかってしまう**ので、一度作成したら `save_to_disk` 関数を使うことで json 形式でローカルに保存しておくことができます。また、ローカルに保存したインデックスを読み込むことも出来ます。

```python
# インデックスの保存
index.save_to_disk("index.json")

# インデックスの読み込み
index = GPTSimpleVectorIndex.load_from_disk("index.json", llm_predictor=ChatGPTLLMPredictor())
```

`query` 関数の引数として文字列を渡すと、クエリを投げることが出来ます。今回はインデックスの作成、読み込み時に `llm_predictor=ChatGPTLLMPredictor()` を引数として与えているため、OpenAI の [Chat completion](https://platform.openai.com/docs/guides/chat/introduction) を使用して返答を得ています。

```python
res = index.query("質問文")
```

これだけで、外部データを反映した返答が得られます。

## 実際に試してみる
最終的なディレクトリ構成は以下のようになります。
```
.
├── data
│   └── blog.txt
├── index
│   └── index.json
└── main.py
```

今回は Fusic の記事の中から、
- [【Fusicで働くとは？vol.14】社会人二年目の4名が振り返る、新卒としての一年間（2022年新卒入社）](https://fusic.co.jp/doings/352)

を使用して ChatGPT にクエリを投げてみたいと思います。

まず始めの準備として、上の記事をコピペし、テキストファイルとして `data/blog.txt` に保存します。LlamaIndex には Web ページをデータとしてインデックスを作れるクラス `SimpleWebPageReader` も提供されているのですが、`html` タグなどがノイズとなるため、今回はテキストファイルとして保存しました。

テキストファイルを保存した後に `main.py` を実行します。コードは以下になります。`OPENAI_API_KEY` を環境変数として定義しておかないといけないので注意してください（`export OPENAI_API_KEY=<ここを自分の API KEY にする>` を実行しておくと楽です）

```python
import os
from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader
from llama_index.llm_predictor.chatgpt import ChatGPTLLMPredictor

def main():
    pwd = __file__
    pwd_dir = pwd.rsplit("/", 1)[0]
    index_dir = os.path.join(pwd_dir, "index")
    index_file = os.path.join(index_dir, "index.json")

    if not os.path.exists(index_file):
        os.makedirs(index_dir, exist_ok=True)

        documents = SimpleDirectoryReader(os.path.join(pwd_dir, "data")).load_data()
        index = GPTSimpleVectorIndex(
            documents=documents, llm_predictor=ChatGPTLLMPredictor()
        )
        index.save_to_disk(index_file)
    else:
        index = GPTSimpleVectorIndex.load_from_disk(
            index_file,
            llm_predictor=ChatGPTLLMPredictor(),
        )
    while True:
        inp = input("聞きたいことを教えてください。\n>>>")
        print(index.query("日本語で答えてください。" + inp))
        print()


if __name__ == "__main__":
    main()
```

上のコードでは、実行するファイルと同じ場所に `data` ディレクトリがあることを想定しています。また `index` ディレクトリを作成し、その中に作成したインデックスを `index.json` として保存します。

実行すると、
```bash
聞きたいことを教えてください。
>>>
```
と出てくると思うので、質問文を入力すれば答えが返ってきます。上のコードの `query` 関数で「日本語で答えてください。」と書いているのは、日本語で質問文を入力しても英語で返ってくることがあるからです。

記事からしか分からない質問を二つ試してみます。
例えば「登場する人物を教えてください」と聞いてみると
```bash
聞きたいことを教えてください。
>>> 登場する人物をリスト形式で教えてください
先進技術本部 先進技術部門 機械学習チーム　ハン ボムソク
技術開発本部 技術開発第二部門　山路 直人
先進技術本部 先進技術部門 機械学習チーム　トシュコフ テオドル
技術開発本部 技術開発第一部門　光岡 郁穂
```

と出力されました。登場人物としては少ない（インタビュアーの佐藤さんがいない）ですが、ChatGPT にはない知識を答えられているという点では成功です。

また、「山路直人について教えてください」と聞いてみると
```bash
聞きたいことを教えてください。
>>> 山路直人について教えてください
山路直人は、フロントエンドもRubyも人一倍勉強していると思われ、社内のMTGでもしっかり発言していて存在感があると述べられています。また 、RubyKaigiのような大きな技術のカンファレンスで登壇したいという目標を持っているとも述べられています。
```
と出力されました。ちゃんとインタビューの内容を元に出力できています！

## GPT-4 を試してみる。
`ChatGPTLLMPredictor` は、デフォルトでは `gpt-3.5-turbo` を使用していますが、最近では GPT-4 が使えることが発表されました。LlamaIndex でも以下のように使用するモデルを指定することで GPT-4 を使用することが出来ます。

```python
from langchain.chat_models import ChatOpenAI
# model_name で `gpt-4` を指定
llm = ChatOpenAI(temperature=0, model_name="gpt-4")
...  # 省略
index = GPTSimpleVectorIndex(
    documents=documents, llm_predictor=ChatGPTLLMPredictor(llm=llm)
)
```

実際に試して、どのように変わるか試してみます。
```bash
聞きたいことを教えてください。
>>> 登場人物をリスト形式で教えてください。
- ハン ボムソク（先進技術本部 先進技術部門 機械学習チーム）
- 山路 直人（技術開発本部 技術開発第二部門）
- トシュコフ テオドル（先進技術本部 先進技術部門 機械学習チーム）
- 光岡 郁穂（技術開発本部 技術開発第一部門）
- 佐藤（採用担当）
```

ちゃんと登場人物が 5 人になっています！ `gpt-3.5-turbo` ではほぼ本文から抜き出したような結果が出力されていたのですが、こちらでは所属が名前の後ろに括弧つきで表示されています。佐藤さんに関しては本文で「採用担当の佐藤」と書かれているだけなのですが、他の人の表示に合わせてリスト形式で書かれており、この記事から分かる情報としては申し分ない結果が出力がされています。

さらに、同じように「山路直人について教えてください」と聞いてみます。
```bash
聞きたいことを教えてください。
>>> 山路直人について教えてください
山路直人さんは、フロントエンドやRubyの技術に熱心に取り組んでおり、社内のMTGでもしっかり発言し、存在感があります。また、元気で体力があり、同僚たちと一緒に最後までやり切る姿が印象的です。今後の目標として、RubyKaigiのような大きな技術カンファレンスで登壇することを目指しています。
```

と出力されました。こちらは `gpt-3.5-turbo` に比べると「元気で体力があり、同僚たちと一緒に最後までやり切る姿が印象的」との情報が追加されています。しかし、これは山路さんが他の三人について言及した内容なので、この記事から分かる山路さんのこととしては不正確です。文体としてはこちらの方が個人的に好きなのですが、これは好みが分かれそうです。

## まとめ
この記事では LlamaIndex について、簡単な使い方を示し、実際に試してみました。自身の持っているデータを参考にして ChatGPT が持っていない知識についても文の生成が出来るので、パーソナライズしたチャットボットなどが手軽に作成できそうです。どうやってローカルのデータを使用しているのかなど、LlamaIndex の詳細な動作については、次回（やる気と時間があれば）書きたいと思います。

## 参考
[LlamaIndex クイックスタートガイド](https://note.com/npaka/n/n8c3867a55837)
