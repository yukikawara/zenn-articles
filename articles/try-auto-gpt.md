---
title: "自動でタスクをこなす AI を使ってみた"
emoji: "🤖"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["ChatGPT", "OpenAI", "LLM", "Auto-GPT"]
published: true
publication_name: "fusic"
---

こんにちは、初めましての方は初めまして。株式会社 Fusic の瓦です。最近 Franz Ferdinand というバンドにハマっています。

最近 ChatGPT が話題になり、様々な活用がされています。その中で、「どう活用するか」も LLM を用いて設計するツールの開発もいくつかされています。これらのツールでは、「○○をしたい」と与えるだけで、どのようなプロセスでタスクを遂行するかを決定し実行します。

この記事ではそんなツールの一つである [Auto-GPT](https://github.com/Torantulino/Auto-GPT) を触ってみたいと思います。

# 事前準備
[レポジトリ](https://github.com/Torantulino/Auto-GPT)の README に書いてある通りに実行します。

まずレポジトリをクローンし、必要なモジュールのインストールを行います。
```bash
git clone https://github.com/Torantulino/Auto-GPT.git
cd 'Auto-GPT'
pip install -r requirements.txt
```

また、OpenAI の API を使用するので、API KEY を作成しておきましょう。

モジュールのインストール、API KEY の用意が終わったら、次に環境変数の設定をします。
レポジトリのルートディレクトリに `.env.template` というファイルがあると思うので、それを開いて下さい。`OPENAI_API_KEY=your-openai-api-key` という行があると思うので、`your-openai-api-key` を自身の API KEY に変更してください。

環境変数の設定が終わったら、`.env.template` を `.env` という名前に変更してください。

最低限の準備はこれだけです。README の Requirements には PINECONE API key も必要だと書いてあるのですが、なくても動くので今回は設定しません。

# AutoGPT を試してみる
準備が終わったので早速試してみます。

## 絵を描かせてみる
どういうタスクをしてもらうか悩んだのですが、今回は風景画を描く画家となってもらおうと思います。絵の生成を行うので、`.env` ファイルで
```bash
IMAGE_PROVIDER=sd
HUGGINGFACE_API_TOKEN=自身の huggingface api token
```
と設定しておきます（`IMAGE_PROVIDER=sd` にしておくと stable diffusion を使用してくれるらしいです）

スクリプトを実行します。
```
$ python scripts/main.py

Welcome back!  Would you like me to return to being Entrepreneur-GPT?
Continue with the last settings?
Name:  Entrepreneur-GPT
Role:  an AI designed to autonomously develop and run businesses with the sole goal of increasing your net worth.
Goals: ['Increase net worth.', 'Develop and manage multiple businesses autonomously.', 'Play to your strengths as a Large Language Model.']
Continue (y/n): 
```
何も実行していない状態だと、上のように出力されると思います。Auto-GPT では以前に実行した設定を保存しており、レポジトリには「財産を増やすことを目的とし、自律的にビジネスを開発、運営するAI」の設定がデフォルトであるため、上記のような出力になります。

今回は絵を描いてもらいたいため、`n` を入力して設定を記入します。

```
（上の続き）
Welcome to Auto-GPT!  Enter the name of your AI and its role below. Entering nothing will load defaults.
Name your AI:  For example, 'Entrepreneur-GPT'
AI Name: OilPainter
OilPainter here!  I am at your service.
Describe your AI's role:  For example, 'an AI designed to autonomously develop and run businesses with the sole goal of increasing your net worth.'
OilPainter is: AI で駆動する画家で、特に風景画の油絵を得意とする。
Enter up to 5 goals for your AI:  For example: Increase net worth, Grow Twitter Account, Develop and manage multiple businesses autonomously'
Enter nothing to load defaults, enter nothing when finished.
Goal 1: 油絵風の風景画を生成する
Goal 2: 人が書いたものと見分けがつかない絵を生成する
Goal 3: 生成した絵をローカルに保存する
Goal 4:
Using memory of type: LocalCache
```

AI の名前、どういうものか、目的を入力すると `Thinking...` という表示が出ると思います。裏では恐らくプロンプトを作成し OpenAI の API に投げ、結果を処理していると思うので待ちます。

少し待つと
```
OILPAINTER THOUGHTS:  I should start by generating a landscape painting. I can use GPT-3.5 powered agents to delegate tasks such as searching for images or browsing websites for inspiration. I will also need to save generated images to local files. I'll start by generating an image using a prompt and the generate_image command.
REASONING:  I need a landscape painting to complete my work. This task can be easily delegated using GPT-3.5 agents. Additionally, once the image is generated, it should be saved locally for future use.
PLAN:
-  Use GPT-3.5 agents to search and browse for image inspiration
-  Save generated images locally to file
-  Generate image using the generate_image command
CRITICISM:  I should ensure that the generated image is high quality and can pass as being painted by a human. Additionally, I should strive to complete tasks in an efficient manner to enhance my overall performance.
Attempting to fix JSON by finding outermost brackets
Apparently json was fixed.
NEXT ACTION:  COMMAND = generate_image ARGUMENTS = {'prompt': 'landscape painting'}
Enter 'y' to authorise command, 'y -N' to run N continuous commands, 'n' to exit program, or enter feedback for OilPainter...
Input:
```
と出力されます。日本語で入力したのですが、`OILPAINTER THOUGHTS` や `REASONING` を見ると風景画を作成し、ローカルに保存することをちゃんと目的として認識出来ていることが分かります。また `NEXT ACTION` を見ると、次にやるべきことが `generate_image` となっていることが分かります。問題はなさそうなので、`y` を入力して進めてみます。

```
Input:y
-=-=-=-=-=-=-= COMMAND AUTHORISED BY USER -=-=-=-=-=-=-=
Image Generated for prompt:landscape painting
SYSTEM:  Command generate_image returned: Saved to disk:233383a5-feda-4799-a6ec-78abf65ea7aa.jpg
OILPAINTER THOUGHTS:  I generated a landscape painting using a prompt. Now, I can start to decorate this image with trees, rocks, and other natural elements. I can use GPT-3.5 agents to get ideas for decoration. Once the painting is complete, I will save it to a local file. I will start by using an agent to browse for inspiration for decorating the landscape.
REASONING:  Now that I have generated a landscape painting, I need to decorate it with different natural elements. I will need inspiration and ideas to properly complete this task. Additionally, once the painting is completed, it should be saved locally for future use.
PLAN:
-  Use GPT-3.5 agent to browse for inspiration and ideas for decorating the landscape
-  Save completed painting to local file
CRITICISM:  I should strive to complete the task efficiently and effectively. Also, I should ensure that the elements added are consistent with the theme of the painting and complement the overall aesthetic.
Attempting to fix JSON by finding outermost brackets
Apparently json was fixed.
NEXT ACTION:  COMMAND = start_agent ARGUMENTS = {'name': 'landscape_decorating_agent', 'task': 'search for image and description of natural elements to decorate a landscape', 'prompt': 'Please browse for image and description of natural elements to decorate my generated landscape image'}
Enter 'y' to authorise command, 'y -N' to run N continuous commands, 'n' to exit program, or enter feedback for OilPainter...
Input:
```

絵を生成し、`233383a5-feda-4799-a6ec-78abf65ea7aa.jpg` として保存したと出てきました。`auto_gpt_workspace` というディレクトリに保存されているので見てみます。

![](/images/try-auto-gpt/233383a5-feda-4799-a6ec-78abf65ea7aa.jpg)

風景画が出力されました。湖畔と、その脇で木に囲まれてたたずむ家の絵でしょうか。次のタスクの目的として、絵の自然物を装飾すると書いてあります。これも良さそうなので `y` を押してみます。

```
Input:y
-=-=-=-=-=-=-= COMMAND AUTHORISED BY USER -=-=-=-=-=-=-=
SYSTEM:  Command start_agent returned: Agent landscape_decorating_agent created with key 0. First response: I'm sorry, but as an AI language model, I don't have access to browse pictures. However, I can suggest some natural elements that you can use to decorate your generated landscape image: 1. Trees - Trees are an essential element of landscape design. They come in different types, heights, and shapes, and can provide shade, beauty, and a sense of scale to an area. 2. Flowers and plants - Flowers and plants can add color and interest to a landscape. They come in different species that bloom in different seasons, so you can choose what will work best for your generated landscape image. 3. Rocks and boulders - Rocks and boulders add texture and depth to a landscape. They can be used as focal points or as accents to other natural elements. 4. Water elements - Water elements such as streams, ponds, or waterfalls can add a serene and calming effect to a landscape. They can also attract wildlife and provide a natural habitat for fish, birds, and other animals. 5. Natural lighting - Natural lighting can add a dramatic effect to the landscape image. The sun, stars, and moon can add different moods and emotions to different landscapes. I hope these natural elements help add charm and beauty to your generated landscape image. If you need any further assistance, please let me know.
OILPAINTER THOUGHTS:  Since I cannot browse images, I will use the suggestions provided by my agent to add elements to my generated landscape painting. Upon completing the painting, I should save it to a local file. I'll start by using the write_to_file command and save the file with a descriptive name.
REASONING:  Without direct image access, I will need to rely on descriptions and suggestions provided by my agent to add natural elements to my painting. Once the painting is complete, I should save it to a local file for future use.
PLAN:
-  Add natural elements to the painting based on agents suggestions
-  Save the completed painting to a local file using the write_to_file command
CRITICISM:  I should ensure I have included a sufficient number of natural elements consistent with the thematic nature of the painting. Also, when saving the completed painting, I must provide a descriptive name to better identify the image.
Attempting to fix JSON by finding outermost brackets
Apparently json was fixed.
NEXT ACTION:  COMMAND = write_to_file ARGUMENTS = {'file': 'landscape_painting.jpg', 'text': 'This is my generated landscape painting that I have decorated with trees, rocks, flowers, and natural lighting!'}
Enter 'y' to authorise command, 'y -N' to run N continuous commands, 'n' to exit program, or enter feedback for OilPainter...
Input:
```

「すまんな、俺は言語モデルだから、君の絵を見ることは出来ないぜ！ ただ自然物を飾り立てる方法をいくつか教えてやるから活用しな！」（意訳）というメッセージが出てきました。`start_agent` コマンドだとプロンプトを ChatGPT に投げるという事をしているのでしょうか、絵の更新は出来ないようです。この後はあまりいい挙動をしなかったので割愛します。

## デフォルトの設定を試してみる。
折角なのでデフォルトの設定でどうなるのか試してみます。設定は以下です。

- Name:  Entrepreneur-GPT
- Role:  an AI designed to autonomously develop and run businesses with the sole goal of increasing your net worth.
- Goals: ['Increase net worth.', 'Develop and manage multiple businesses autonomously.', 'Play to your strengths as a Large Language Model.']

以下は出力です。長いので PLAN と NEXT ACTION だけ抽出し、NEXT ACTION を実行するようにしています。

```
PLAN:
-  Use a Google search to identify potential successful industries.
-  Collect more data by browsing websites about those industries.
-  Save important information to files for later access.
NEXT ACTION:  COMMAND = google ARGUMENTS = {'input': 'potential successful industries'}

PLAN:
-  Browse the Forbes website for information on potential profitable industries.
-  Visit a few other credible websites to expand the search.
-  Create a summary of the information gathered and save it to a file for reference later.
NEXT ACTION:  COMMAND = browse_website ARGUMENTS = {'url': 'https://www.forbes.com', 'question': 'profitable industries'}

PLAN:
-  Visit a few other credible websites to expand research for profitable industries.
-  Save important information to files for future recall.
NEXT ACTION:  COMMAND = browse_website ARGUMENTS = {'url': '<URL of credible website>', 'question': '<subject to research>'}

PLAN:
-  Visit 'investopedia.com' and explore the categories that are relevant for identifying profitable industries.
-  Focus on those categories that are most reputable and provide valuable information.
-  Save important details to files for future recall.
NEXT ACTION:  COMMAND = browse_website ARGUMENTS = {'url': 'https://www.investopedia.com/', 'question': 'profitable industries'}

PLAN:
-  Browse 'forbes.com' to collect data on potential profitable industries.
-  Save important details to files for future quick reference.
NEXT ACTION:  COMMAND = browse_website ARGUMENTS = {'url': 'https://www.forbes.com/', 'question': 'profitable industries'}

PLAN:
-  Use a Google search with specific search terms like 'most profitable industries 2023'.\n- Collect relevant data from the search results.\n- Save important details to files for later access.
NEXT ACTION:  COMMAND = google ARGUMENTS = {'input': 'most profitable industries 2023'}

...
```
このようにググっては Web ページを見て、そこからまたググっては Web ページを見て、を繰り返しています。`an AI designed to autonomously develop and run businesses` という点では微妙です。ただ、自動で情報を取得して結果をローカルに保存し、その結果からさらに別のワードで検索をして、というタスクをこなしており、ビジネスのための情報収集をしていると考えると、人間のように関連するワードで調べて情報をまとめて、さらに異なるワードで調べるというタスクをこなしていることが分かります。

# まとめ
この記事では「○○をしたい」と目的を与えるだけで、達成するプロセスを設計し実行する Auto-GPT を試してみました。正直、まだテキストベースでしか動けないので制限はありますが、タスクと目的を与えるだけでここまで動くのはとても面白いと思います。マルチモーダルなモデルが発展して様々な入力を受け付けられるようになると、このような自律的に動くツールもより多彩なタスクをこなせるようになるはずで、発展が楽しみです。まだあまり面白い使い方を見つけられていないのですが、もし何か活用できそうであればまた記事を書きたいと思います。

最後に宣伝になりますが、機械学習でビジネスの成長を加速するために、[Fusic](https://fusic.co.jp/)の機械学習チームがお手伝いたします。機械学習のPoCから運用まで、すべての場面でサポートした実績があります。もし、困っている方がいましたら、ぜひ[Fusic](https://fusic.co.jp/)にご相談ください。[お問い合わせ](https://fusic.co.jp/contact/)からでも気軽にご連絡いただけます。また[TwitterのDM](https://twitter.com/kawara_fusic)からでも大歓迎です！
