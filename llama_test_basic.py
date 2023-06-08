#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 20 09:36:44 2023

@author: kbillesk
"""

from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, download_loader, GPTRAKEKeywordTableIndex
from llama_index import LLMPredictor, ServiceContext, StorageContext, load_index_from_storage
filename_fn = lambda filename: {'file_name': filename}

service_context = ServiceContext.from_defaults()
documents = SimpleDirectoryReader('examples/gdpr_expert/data', file_metadata=filename_fn).load_data()
#The following two lines are used to build/update the vectorstore index.
#If you dont need this comment out these lines and activate two susequent lines, which reads from the existing index
index = GPTVectorStoreIndex.from_documents(documents,service_context=service_context)
index.storage_context.persist(persist_dir="examples/gdpr_expert/storage")
#storage_context = StorageContext.from_defaults(persist_dir="example/gdpr_expert/storage")
#index = load_index_from_storage(storage_context)
query_engine = index.as_query_engine(service_context=service_context)
#openai_api_key="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"
question = "ww"
while question != "quit":
    question = input("Stil dit GDPR spørgsmål?\n")
    if question != "quit":
        response = query_engine.query(question)
        print(response)
        for node in response.source_nodes:
            print('-----')
            text_fmt = node.node.text.strip().replace('\n', ' ')[:1000]
            doc_data = node.node.get_doc_hash()
            #print(f"Text:\t {text_fmt} ...")
            print(f'Metadata:\t {node.node.ref_doc_id}')
            print(f'Metadata:\t {node.node.extra_info_str}')
            print(doc_data)
            print(f'Score:\t {node.score:.3f}')
            #print(response)