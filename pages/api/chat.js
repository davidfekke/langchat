import { streamToResponse } from 'ai';

import { ChatOllama } from "@langchain/community/chat_models/ollama";
import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama";
import { PGVectorStore } from "@langchain/community/vectorstores/pgvector";
import { formatDocumentsAsString } from "langchain/util/document";
import { PromptTemplate } from "@langchain/core/prompts";
import {
  RunnableSequence,
  RunnablePassthrough,
} from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";

import * as dotenv from "dotenv";
dotenv.config();

const embeddings = new OllamaEmbeddings({
    model: "nomic-embed-text", // default value
    baseUrl: "http://localhost:11434", // default value
    requestOptions: {
        useMMap: true,
        numThread: 6,
        numGpu: 1,
    },
});

const pgconfig = {
    postgresConnectionOptions: {
      type: "postgres",
      host: "localhost",
      port: 5432,
      user: "postgres",
      password: "password",
      database: "pragmaticvector",
    },
    tableName: "vectordocs",
    columns: {
      idColumnName: "id",
      vectorColumnName: "vector",
      contentColumnName: "content",
      metadataColumnName: "metadata",
    },
};

export const config = {
    api: {
      bodyParser: true,
    }
};

export default async function handler(req, res) {
    const body = await req.body.query;

    const pgvectorStore = await PGVectorStore.initialize(
        embeddings,
        pgconfig
    );    

    try {
        const model = new ChatOllama({
            baseUrl: "http://localhost:11434", // Default value
            model: "llama3", // Default value
            debug: false
        });
    
        const retriever = pgvectorStore.asRetriever();

        const prompt = PromptTemplate.fromTemplate(`Answer the question based only on the following context:
        {context}
        
        Question: {question}`);
        
        const chain = RunnableSequence.from([
          {
              context: retriever.pipe(formatDocumentsAsString),
              question: new RunnablePassthrough(),
          },
          prompt,
          model,
          new StringOutputParser(),
        ]);

        const stream2 = await chain.stream(body);
        
        streamToResponse(stream2, res)
        pgvectorStore.end();
    } catch (error) {
        console.error(error);
        res.status(500).send("Internal Server Error");
        pgvectorStore.end();
        return new Response(
            JSON.stringify(
                { error: error.message },
                {
                    status: 500,
                    headers: { "Content-Type": "application/json" },
                }
            )
        );
    }
}