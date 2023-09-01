#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 13:55:25 2023

@author: kbillesk
"""
import nltk
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index import ServiceContext, SimpleDirectoryReader, VectorStoreIndex, Prompt, PromptHelper, StorageContext, load_index_from_storage
from llama_index.node_parser import SimpleNodeParser, SentenceWindowNodeParser
from llama_index.llms import OpenAI
from llama_index.llms import openai_utils
from llama_index.indices.postprocessor import MetadataReplacementPostProcessor
print("pet")
node_parser = SentenceWindowNodeParser.from_defaults(window_size=3)
llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
#service_context = ServiceContext.from_defaults(llm=llm, embed_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"), node_parser=node_parser)
service_context = ServiceContext.from_defaults(llm=llm, node_parser=node_parser)
print(service_context)
template = (
    "Brug følgende information som kontekst. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Svar på følgende spørgsmål på baggrund af denne information: {query_str}\n. Giv et uddybet svar. Giv svaret på dansk."
)
qa_template = Prompt(template)
documents = SimpleDirectoryReader('legaldata/').load_data()
#index = VectorStoreIndex.from_documents(documents,service_context=service_context)
#index.storage_context.persist(persist_dir="legalstorage")
storage_context = StorageContext.from_defaults(persist_dir="legalstorage")
index = load_index_from_storage(storage_context)
query_engine = index.as_query_engine(similarity_top_k=5, node_postprocessors=[MetadataReplacementPostProcessor(target_metadata_key="window")],text_qa_template=qa_template)
question = "ww"
while question != "quit":
    question = input("Stil dit spørgsmål til lejeloven?\n")
    if question != "quit":
        response = query_engine.query(question)
        topnode = response.source_nodes[0] 
        if  topnode.score > 0.82:
            print(response)
            for node in response.source_nodes:
                print('-----')
                text_fmt = node.node.text.strip().replace('\n', ' ')[:1000]
                node_parent = "NONE"
                if hasattr(node, 'parent_node_id'):
                    node_parent = node.node.parent_node_id
                node_content = node.node.get_text()
                node_filename = node.node.get_metadata_str('file_name')
                print('filename: '+str(node_filename))
                print(f"Text:\t {text_fmt} ...")
                print(f'doc_id:\t {node.node.ref_doc_id}')
                #print("node text: "+node_content)
                print("Node parent: "+node_parent)
                #print(f'Metadata:\t {node.node.get_metadata_str}')
                print(f'Score:\t {node.score:.3f}')
                #print(response)
        else:
            print("Spørgsmålet ligger udenfor modellens kompetence.")
        