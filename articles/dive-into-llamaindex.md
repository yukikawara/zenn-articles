---
title: "LlamaIndex で ChatGPT に専門知識を組み込む裏側"
emoji: "🦙"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["OpenAI", "ChatGPT", "LlamaIndex"]
published: true
publication_name: "fusic"
---

こんにちは、初めましての方は初めまして。株式会社 Fusic で機械学習エンジニアをしている瓦です。最近は花粉との戦いで連日連夜盛り上がっています。

前回、[LlamaIndex で ChatGPT に専門知識を組み込んでみた](https://zenn.dev/fusic/articles/try-llamaindex) でローカルのデータに基づいて ChatGPT を使用する方法を紹介しました。ChatGPT には存在しないローカルのデータに基づいて文を生成出来ていましたが、LlamaIndex では裏でどのようにローカルのデータを使用しているのでしょうか？ この記事では、LlamaIndex がどのように動いているのかを探っていきたいと思います[^1]。

[^1]: この記事を書いた時点での LlamaIndex のバージョンは `0.5.6` です。バージョンによっては実装が異なるかもしれません。

## 作成されたインデックスの確認
とりあえず、LlamaIndex で生成されるインデックスについて見ていきましょう。簡単のために、「今日の晩御飯はハンバーグ！」とだけ書いたテキストファイルを用意し、LlamaIndex を使用してみます。

実行したコードは以下になります。

```python
from langchain.chat_models import ChatOpenAI
from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader
from llama_index.llm_predictor.chatgpt import ChatGPTLLMPredictor
from llama_index.indices.service_context import ServiceContext


def main():
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    service_context = ServiceContext.from_defaults(
        llm_predictor=ChatGPTLLMPredictor(llm=llm)
    )

    documents = SimpleDirectoryReader("./data").load_data()
    index = GPTSimpleVectorIndex.from_documents(
        documents, service_context=service_context
    )
    index.save_to_disk("index.json")

    print(index.query(input(), similarity_top_k=1))

if __name__ == "__main__":
    main()
```

このコードを実行すると、`data` ディレクトリに保存されているファイルを使用してインデックスが構築され、`index.json` というファイルに保存されます。

どのようなインデックスが保存されているのか確認するため、作成された json ファイルの中身を見てみます。
```json
{
    "index_struct": {
        "__type__": "simple_dict",
        "__data__": {
            "index_id": "96310bf4-49ae-4770-bb92-ccc41561b99a",
            "summary": null,
            "nodes_dict": {
                "974d47a7-e1ed-4023-b1ec-7a07b3840b15": "2b7cd68a-6cb7-4fd6-80fe-2bf04a60f4ac"
            },
            "doc_id_dict": {
                "a783d137-512a-46cf-8152-6f8cdf82408f": [
                    "974d47a7-e1ed-4023-b1ec-7a07b3840b15"
                ]
            },
            "embeddings_dict": {
                "974d47a7-e1ed-4023-b1ec-7a07b3840b15": [
                    0.0058559877797961235,
                    0.009936033748090267,
                    ...
                    -0.003624506527557969
                ]
            }
        }
    },
    "docstore": {
        "docs": {
            "2b7cd68a-6cb7-4fd6-80fe-2bf04a60f4ac": {
                "text": "\u4eca\u65e5\u306e\u6669\u5fa1\u98ef\u306f\u30cf\u30f3\u30d0\u30fc\u30b0\uff01",
                "doc_id": "2b7cd68a-6cb7-4fd6-80fe-2bf04a60f4ac",
                "embedding": null,
                "doc_hash": "02bd0d753811929ebfa1ac35204aa5a089e58e2d0291c7aa7afae85aa34b806b",
                "extra_info": null,
                "node_info": {
                    "start": 0,
                    "end": 13
                },
                "relationships": {
                    "1": "a783d137-512a-46cf-8152-6f8cdf82408f"
                },
                "__type__": "1"
            }
        },
        "ref_doc_info": {
            "a783d137-512a-46cf-8152-6f8cdf82408f": {
                "doc_hash": "02bd0d753811929ebfa1ac35204aa5a089e58e2d0291c7aa7afae85aa34b806b"
            },
            "2b7cd68a-6cb7-4fd6-80fe-2bf04a60f4ac": {
                "doc_hash": "02bd0d753811929ebfa1ac35204aa5a089e58e2d0291c7aa7afae85aa34b806b"
            }
        }
    },
    "vector_store": {
        "simple_vector_store_data_dict": {
            "embedding_dict": {
                "974d47a7-e1ed-4023-b1ec-7a07b3840b15": [
                    0.0058559877797961235,
                    0.009936033748090267,
                    ...
                    -0.003624506527557969
                ]
            },
            "text_id_to_doc_id": {
                "974d47a7-e1ed-4023-b1ec-7a07b3840b15": "a783d137-512a-46cf-8152-6f8cdf82408f"
            }
        }
    }
}
```

`index_struct`, `docstore`, `vector_store` という三つのキーがあり、それぞれ情報が格納されています。`index_struct` にはテキストと得られたベクトル表現を紐づける情報が、`docstore` にはテキストとそれぞれの ID が、`vector_store` には得られたベクトル表現とそれぞれの ID が格納されています。

LlamaIndex では

1. ローカルのデータをベクトル表現に変換
2. クエリが投げられるとベクトル表現を使用して類似した文書を探索
3. 適切にプロンプトに組み込んで文を生成

を行うことで、ローカルのデータに基づいた文を生成していそうです。


## Dive into LlamaIndex
上記で見たように、LlamaIndex では大きく分けて

1. インデックスの構築
2. 類似文の検索
3. 文の生成

の三つの動作を行っています。2. と 3. は一続きの動作として見た方が分かりやすいと思うので、「[インデックスの構築](#インデックスの構築)」、「[類似文の検索および文の生成](#類似文の検索および文の生成)」の二つに分けて詳細を見ていきます。

### インデックスの構築
与えられたテキストから、検索用のインデックスを構築します。LlamaIndex にはインデックスの作成方法として `List Index`, `Vector Store Index`, `Tree Index` の三通り用意されています（[LlamaIndex のドキュメント](https://gpt-index.readthedocs.io/en/latest/guides/primer/index_guide.html)）
今回は `GPTSimpleVectorIndex` を使用しているので、`Vector Store Index` を見ていきます。

`Vector Store Index` では、`Node` と呼ばれるものにテキストとそのベクトル表現が格納されています（ページの上に書いてあるように、`Node` は実際には「チャンク」と呼ばれる、テキストをより細かく分割したものと対応しています。そのため、`Node` には、テキスト全体ではなくテキストの断片的な情報が含まれていると思った方がいいです）つまり、

1. テキストを読み込む
2. テキストを一定の長さで分割し、それぞれに対応した `Node` を作成
3. それぞれのテストのベクトル表現を取得

という流れで各テキストとそのベクトル表現の対応関係を得て、インデックスを構築しています。

実際にインデックスを作成してから OpenAI API の Usage のページを見ると、以下のように embedding の API が叩かれていることが確認できます。

![](/images/dive-into-llamaindex/embedding_usage.png)

ここでは `text-embedding-ada-002-v2` というモデルでテキストに対応したベクトルを取得していることが分かります。ちなみにテキストを分ける長さは

```python:gpt_index/contants.py
MAX_CHUNK_SIZE = 3900
```
とあります。おそらく質問文を加えても GPT-3.5 系の API の制限長である 4096 に届かない長さにしているのではないかと思います（詳しい方がいれば教えてください）

### 類似文の検索および文の生成
インデックスを構築した後は、取得したベクトル表現をもとに類似した文の検索を行い、プロンプトに含めて ChatGPT へとクエリを投げます。

まず類似した文の検索についてですが、これは

1. クエリをベクトル表現へと変換
2. インデックスのベクトル表現と類似度を計算し、近いものを選択

という流れで行われています。ベクトル表現への変換はインデックスの構築と同様に OpenAI の API を使用して得ています。類似度については

```python:gpt_index/embeddings/base.py
def similarity(
    embedding1: EMB_TYPE,
    embedding2: EMB_TYPE,
    mode: SimilarityMode = SimilarityMode.DEFAULT,
) -> float:
    """Get embedding similarity."""
    if mode == SimilarityMode.EUCLIDEAN:
        return float(np.linalg.norm(np.array(embedding1) - np.array(embedding2)))
    elif mode == SimilarityMode.DOT_PRODUCT:
        product = np.dot(embedding1, embedding2)
        return product
    else:
        product = np.dot(embedding1, embedding2)
        norm = np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        return product / norm
```

に実装があります。指定がなければコサイン類似度を使用して類似度の計算を行っているようです。また、クエリを投げるときにデフォルトでは類似度が高いものを一つだけ使用していますが、`query` 関数の引数 `similarity_top_k` を変えることでより多くの文を取ってくることが出来ます。

次にプロンプトに含める部分です。クエリの前処理の実装が以下になります。

```python:gpt_index/indices/vector_store/base.py
from gpt_index.prompts.default_prompts import DEFAULT_TEXT_QA_PROMPT
...
        self.text_qa_template = text_qa_template or DEFAULT_TEXT_QA_PROMPT
...
    def _preprocess_query(self, mode: QueryMode, query_kwargs: Any) -> None:
        super()._preprocess_query(mode, query_kwargs)
        if "text_qa_template" not in query_kwargs:
            query_kwargs["text_qa_template"] = self.text_qa_template
        # NOTE: Pass along vector store instance to query objects
        # TODO: refactor this to be more explicit
        query_kwargs["vector_store"] = self._vector_store
```

指定がなければ、プロンプトのテンプレートとして `DEFAULT_TEXT_QA_PROMPT` というものが使用されているようです。この変数が定義されている箇所を見てみましょう。

```python:gpt_index/prompts/default_prompts.py
DEFAULT_TEXT_QA_PROMPT_TMPL = (
    "Context information is below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the question: {query_str}\n"
)
DEFAULT_TEXT_QA_PROMPT = QuestionAnswerPrompt(DEFAULT_TEXT_QA_PROMPT_TMPL)
```

つまり検索で抽出できた文を `context_str` に埋め込み、「文脈に沿って答えてください」と質問を投げているようです。こうすることで ChatGPT では学習出来ていなかった情報についても回答できる仕組みになっています。

## まとめ
以上のようにして、LlamaIndex ではローカルのデータに基づいて ChatGPT を使用しています。仕組み自体は思っているよりも簡単ですが、これらの実装をより簡単に使用できるライブラリとして提供しているのは素晴らしいですね。今回は紹介を省きましたが、`SimpleDirectoryReader` では色々なファイル形式に対応出来るよう実装されており、活用の幅が広がりそうです。

最後に宣伝になりますが、機械学習でビジネスの成長を加速するために、[Fusic](https://fusic.co.jp/)の機械学習チームがお手伝いたします。機械学習のPoCから運用まで、すべての場面でサポートした実績があります。もし、困っている方がいましたら、ぜひ[Fusic](https://fusic.co.jp/)にご相談ください。[お問い合わせ](https://fusic.co.jp/contact/)から気軽にご連絡いただけますが、[TwitterのDM](https://twitter.com/kawara_fusic)からでも大歓迎です！
