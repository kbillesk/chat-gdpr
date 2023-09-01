from llama_index import ServiceContext, SimpleDirectoryReader, VectorStoreIndex, Prompt, PromptHelper, StorageContext, load_index_from_storage
from llama_index.node_parser import SimpleNodeParser, SentenceWindowNodeParser
from llama_index.llms import OpenAI
from llama_index.llms import openai_utils
from llama_index.indices.postprocessor import MetadataReplacementPostProcessor
from llama_index.text_splitter import SentenceSplitter
print("pet")
node_parser = SentenceWindowNodeParser.from_defaults(window_size=3)
llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
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
#Replace with directory where you have documents that should be indexed
documents = SimpleDirectoryReader('data/').load_data()
#The following two lines should be used when building the vectorstore. If you have the vectorstore already, comment out the two lines and uncomment the next two
index = VectorStoreIndex.from_documents(documents,service_context=service_context, show_progress=True)
index.storage_context.persist(persist_dir="sentstorage")
#storage_context = StorageContext.from_defaults(persist_dir="sentstorage")
#index = load_index_from_storage(storage_context)
query_engine = index.as_query_engine(similarity_top_k=3, node_postprocessors=[MetadataReplacementPostProcessor(target_metadata_key="window")],text_qa_template=qa_template)
#The rest is just a simple query interface. It juses a threshold for the score (0.82) this needs to be set otherwise it will respond to all kinds of questions.
question = "ww"
while question != "quit":
    question = input("Stil dit GDPR spørgsmål?\n")
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
            print("Spørgsmålet ligger udenfor chat-gdprs kompetence.")
        
