---
title: "LlamaIndex"
emoji: "🦙"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["OpenAI", "ChatGPT", "LlamaIndex"]
published: false
---

こんにちは、初めましての方は初めまして。株式会社 Fusic で機械学習エンジニアをしている瓦です。最近は花粉との戦いで連日連夜盛り上がっています。

前回、[LlamaIndex で ChatGPT に専門知識を組み込んでみた](https://zenn.dev/fusic/articles/try-llamaindex) でローカルのデータに基づいて ChatGPT を使用する方法を紹介しました。結果を見てみるとちゃんと与えたデータに基づいて文を生成出来ているようですが、LlamaIndex では裏でどのようにローカルのデータを使用しているのでしょうか？ この記事では、実装を見ながら LlamaIndex がどのように動いているのかを探っていきたいと思います。

## 作成されたインデックスの確認
とりあえず、LlamaIndex で生成されるインデックスについて色々見ていきましょう。簡単のために、「今日の晩御飯はハンバーグ！」とだけ書いたテキストファイルを用意し、LlamaIndex を使用してみます。

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

    # vvvvv 1. 与えられたデータからインデックスを構築 vvvvv
    documents = SimpleDirectoryReader("./data").load_data()
    index = GPTSimpleVectorIndex.from_documents(
        documents, service_context=service_context
    )
    index.save_to_disk("index.json")

    # vvvvv 2. 与えられた文字列に似ているチャンクをインデックスから検索して文の生成 vvvvv
    print(index.query(input(), similarity_top_k=1))

if __name__ == "__main__":
    main()
```

このコードを実行すると、`data` ディレクトリに保存されているファイルを基に構築されたインデックスが、`index.json` というファイルに保存されます。

どのようなインデックスが保存されているのか確認するため、作成された json ファイルの中身を見てみましょう。
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

`index_struct`, `docstore`, `vector_store` という三つのキーがあり、それぞれの情報が格納されています。`index_struct` にはテキストと得られたベクトル表現を紐づける情報が、`docstore` にはテキストとそれぞれの ID が、`vector_store` には得られたベクトル表現とそれぞれの ID が格納されています。

LlamaIndex では

1. ローカルのデータをベクトル表現に変換
2. クエリが投げられるとベクトル表現を使用して類似した文書を探索
3. 適切にプロンプトに組み込んで文を生成

を行い、ローカルのデータに基づいた回答を行うことが出来るという仕組みであることが推測できます。


## LlamaIndex の
上記で見たように、LlamaIndex では大きく分けて

1. インデックスの構築
2. 類似文の検索
3. 文の生成

の三つの動作を行っています。
2. と 3. は一続きの動作として見た方が分かりやすいと思うので、「インデックスの構築」、「類似文の検索および文の生成」の二つに分けて以下で詳細を見ていきます。

### インデックスの構築
与えられたテキストから、検索用のインデックスを構築します。LlamaIndex にはインデックスの作成方法として `List Index`, `Vector Store Index`, `Tree Index` の三通り用意されています（[参考リンク](https://gpt-index.readthedocs.io/en/latest/guides/primer/index_guide.html)）
今回は `GPTSimpleVectorIndex` を使用しているので、`Vector Store Index` を見ていきます。

`Vector Store Index` では、`Node` と呼ばれるものにテキストとそのベクトル表現が格納されています。ページの上に書いてあるように、`Node` は実際には「チャンク」と呼ばれる、テキストをより細かく分割したものと対応しています。そのため、`Node` には、テキスト全体ではなくテキストの断片的な情報が含まれていると思った方がいいです。


### 類似文の検索および文の生成
