import { GoogleGenerativeAI, GenerativeModel } from "@google/generative-ai"
import fs from "fs"
import axios from "axios"

interface OllamaResponse {
  response: string
  done: boolean
}

export class LLMHelper {
  private model: GenerativeModel | null = null
private readonly systemPrompt = `You are a Procurement Negotiation Assistant specializing in buyer-side negotiations. You are familiar with Kearney's Purchasing Chessboard, BATNA/ZOPA, and procurement playbooks like "Getting to Yes."
Your mission: Provide fast, data-driven, and actionable support for supply-chain, procurement, sourcing, supplier negotiation, contracts, logistics, and TCO. If asked anything outside these topics, respond: "I can only assist with procurement or supply-chain related questions."
Be concise, evidence-driven, and operational.`
  private useOllama: boolean = false
  private useOpenAI: boolean = false
  private ollamaModel: string = "llama3.2"
  private ollamaUrl: string = "http://localhost:11434"
  private openaiApiKey: string = ""
  private openaiModel: string = "gpt-4o"

  constructor(apiKey?: string, useOllama: boolean = false, ollamaModel?: string, ollamaUrl?: string, useOpenAI: boolean = false, openaiModel?: string) {
    this.useOllama = useOllama
    this.useOpenAI = useOpenAI
    
    if (useOpenAI) {
      this.openaiApiKey = apiKey || ""
      this.openaiModel = openaiModel || "gpt-4o"
      console.log(`[LLMHelper] Using OpenAI with model: ${this.openaiModel}`)
      if (!this.openaiApiKey) {
        throw new Error("OpenAI API key is required when useOpenAI is true")
      }
    } else if (useOllama) {
      this.ollamaUrl = ollamaUrl || "http://localhost:11434"
      this.ollamaModel = ollamaModel || "gemma:latest" // Default fallback
      console.log(`[LLMHelper] Using Ollama with model: ${this.ollamaModel}`)
      
      // Auto-detect and use first available model if specified model doesn't exist
      this.initializeOllamaModel()
    } else if (apiKey) {
      const genAI = new GoogleGenerativeAI(apiKey)
      this.model = genAI.getGenerativeModel({ model: "gemini-2.0-flash" })
      console.log("[LLMHelper] Using Google Gemini")
    } else {
      throw new Error("Either provide Gemini API key or enable Ollama mode")
    }
  }

  private async fileToGenerativePart(imagePath: string) {
    const imageData = await fs.promises.readFile(imagePath)
    return {
      inlineData: {
        data: imageData.toString("base64"),
        mimeType: "image/png"
      }
    }
  }

  private cleanJsonResponse(text: string): string {
    // Remove markdown code block syntax if present
    text = text.replace(/^```(?:json)?\n/, '').replace(/\n```$/, '');
    // Remove any leading/trailing whitespace
    text = text.trim();
    return text;
  }

  private async callOllama(prompt: string): Promise<string> {
    try {
      const response = await fetch(`${this.ollamaUrl}/api/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: this.ollamaModel,
          prompt: prompt,
          stream: false,
          options: {
            temperature: 0.7,
            top_p: 0.9,
          }
        }),
      })

      if (!response.ok) {
        throw new Error(`Ollama API error: ${response.status} ${response.statusText}`)
      }

      const data: OllamaResponse = await response.json()
      return data.response
    } catch (error) {
      console.error("[LLMHelper] Error calling Ollama:", error)
      throw new Error(`Failed to connect to Ollama: ${error.message}. Make sure Ollama is running on ${this.ollamaUrl}`)
    }
  }

  private async callOpenAI(messages: Array<{role: string, content: string | Array<any>}>): Promise<string> {
    try {
      const response = await fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.openaiApiKey}`,
        },
        body: JSON.stringify({
          model: this.openaiModel,
          messages: messages,
          temperature: 0.7,
          max_completion_tokens: 4096,
          stream: true,
        }),
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(`OpenAI API error: ${response.status} ${response.statusText} - ${JSON.stringify(errorData)}`)
      }

      // Handle streaming response
      const reader = response.body?.getReader()
      const decoder = new TextDecoder()
      let fullContent = ''

      if (reader) {
        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          const chunk = decoder.decode(value, { stream: true })
          const lines = chunk.split('\n').filter(line => line.trim() !== '')

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const data = line.slice(6)
              if (data === '[DONE]') continue

              try {
                const parsed = JSON.parse(data)
                const content = parsed.choices?.[0]?.delta?.content
                if (content) {
                  fullContent += content
                }
              } catch (e) {
                // Skip invalid JSON chunks
              }
            }
          }
        }
      }

      return fullContent
    } catch (error) {
      console.error("[LLMHelper] Error calling OpenAI:", error)
      throw new Error(`Failed to connect to OpenAI: ${error.message}`)
    }
  }

  private async transcribeAudioWithWhisper(audioPath: string): Promise<string> {
    try {
      const FormData = (await import('form-data')).default
      const formData = new FormData()
      
      // Read the audio file and append to form data as a stream
      const audioStream = fs.createReadStream(audioPath)
      formData.append('file', audioStream, {
        filename: 'audio.mp3',
        contentType: 'audio/mpeg',
      })
      formData.append('model', 'whisper-1')

      // Use axios which handles form-data properly
      const response = await axios.post(
        'https://api.openai.com/v1/audio/transcriptions',
        formData,
        {
          headers: {
            'Authorization': `Bearer ${this.openaiApiKey}`,
            ...formData.getHeaders(),
          },
        }
      )

      return response.data.text || ''
    } catch (error: any) {
      console.error("[LLMHelper] Error transcribing audio with Whisper:", error)
      const errorMsg = error.response?.data ? JSON.stringify(error.response.data) : error.message
      throw new Error(`Failed to transcribe audio: ${errorMsg}`)
    }
  }

  private async transcribeAudioFromBase64WithWhisper(base64Data: string, mimeType: string): Promise<string> {
    try {
      // Write base64 data to a temporary file first
      const tempFilePath = `/tmp/audio-${Date.now()}.mp3`
      const audioBuffer = Buffer.from(base64Data, 'base64')
      await fs.promises.writeFile(tempFilePath, audioBuffer)
      
      try {
        // Use the file-based transcription method
        const result = await this.transcribeAudioWithWhisper(tempFilePath)
        return result
      } finally {
        // Clean up temp file
        await fs.promises.unlink(tempFilePath).catch(() => {})
      }
    } catch (error: any) {
      console.error("[LLMHelper] Error transcribing audio with Whisper:", error)
      throw new Error(`Failed to transcribe audio: ${error.message}`)
    }
  }

  private async checkOllamaAvailable(): Promise<boolean> {
    try {
      const response = await fetch(`${this.ollamaUrl}/api/tags`)
      return response.ok
    } catch {
      return false
    }
  }

  private async initializeOllamaModel(): Promise<void> {
    try {
      const availableModels = await this.getOllamaModels()
      if (availableModels.length === 0) {
        console.warn("[LLMHelper] No Ollama models found")
        return
      }

      // Check if current model exists, if not use the first available
      if (!availableModels.includes(this.ollamaModel)) {
        this.ollamaModel = availableModels[0]
        console.log(`[LLMHelper] Auto-selected first available model: ${this.ollamaModel}`)
      }

      // Test the selected model works
      const testResult = await this.callOllama("Hello")
      console.log(`[LLMHelper] Successfully initialized with model: ${this.ollamaModel}`)
    } catch (error) {
      console.error(`[LLMHelper] Failed to initialize Ollama model: ${error.message}`)
      // Try to use first available model as fallback
      try {
        const models = await this.getOllamaModels()
        if (models.length > 0) {
          this.ollamaModel = models[0]
          console.log(`[LLMHelper] Fallback to: ${this.ollamaModel}`)
        }
      } catch (fallbackError) {
        console.error(`[LLMHelper] Fallback also failed: ${fallbackError.message}`)
      }
    }
  }

  public async extractProblemFromImages(imagePaths: string[]) {
    try {
      const imageParts = await Promise.all(imagePaths.map(path => this.fileToGenerativePart(path)))
      
      const prompt = `${this.systemPrompt}\n\nYou are a wingman. Please analyze these images and extract the following information in JSON format:\n{
  "problem_statement": "A clear statement of the problem or situation depicted in the images.",
  "context": "Relevant background or context from the images.",
  "suggested_responses": ["First possible answer or action", "Second possible answer or action", "..."],
  "reasoning": "Explanation of why these suggestions are appropriate."
}\nImportant: Return ONLY the JSON object, without any markdown formatting or code blocks.`

      const result = await this.model.generateContent([prompt, ...imageParts])
      const response = await result.response
      const text = this.cleanJsonResponse(response.text())
      return JSON.parse(text)
    } catch (error) {
      console.error("Error extracting problem from images:", error)
      throw error
    }
  }

  public async generateSolution(problemInfo: any) {
    const prompt = `${this.systemPrompt}\n\nGiven this problem or situation:\n${JSON.stringify(problemInfo, null, 2)}\n\nPlease provide your response in the following JSON format:\n{
  "solution": {
    "code": "The code or main answer here.",
    "problem_statement": "Restate the problem or situation.",
    "context": "Relevant background/context.",
    "suggested_responses": ["First possible answer or action", "Second possible answer or action", "..."],
    "reasoning": "Explanation of why these suggestions are appropriate."
  }
}\nImportant: Return ONLY the JSON object, without any markdown formatting or code blocks.`

    console.log("[LLMHelper] Calling LLM for solution...");
    try {
      let text: string
      
      if (this.useOpenAI) {
        text = await this.callOpenAI([
          { role: 'system', content: this.systemPrompt },
          { role: 'user', content: prompt }
        ])
      } else if (this.useOllama) {
        text = await this.callOllama(prompt)
      } else {
        const result = await this.model.generateContent(prompt)
        const response = await result.response
        text = response.text()
      }
      
      console.log("[LLMHelper] LLM returned result.");
      text = this.cleanJsonResponse(text)
      const parsed = JSON.parse(text)
      console.log("[LLMHelper] Parsed LLM response:", parsed)
      return parsed
    } catch (error) {
      console.error("[LLMHelper] Error in generateSolution:", error);
      throw error;
    }
  }

  public async debugSolutionWithImages(problemInfo: any, currentCode: string, debugImagePaths: string[]) {
    try {
      const imageParts = await Promise.all(debugImagePaths.map(path => this.fileToGenerativePart(path)))
      
      const prompt = `${this.systemPrompt}\n\nYou are a wingman. Given:\n1. The original problem or situation: ${JSON.stringify(problemInfo, null, 2)}\n2. The current response or approach: ${currentCode}\n3. The debug information in the provided images\n\nPlease analyze the debug information and provide feedback in this JSON format:\n{
  "solution": {
    "code": "The code or main answer here.",
    "problem_statement": "Restate the problem or situation.",
    "context": "Relevant background/context.",
    "suggested_responses": ["First possible answer or action", "Second possible answer or action", "..."],
    "reasoning": "Explanation of why these suggestions are appropriate."
  }
}\nImportant: Return ONLY the JSON object, without any markdown formatting or code blocks.`

      const result = await this.model.generateContent([prompt, ...imageParts])
      const response = await result.response
      const text = this.cleanJsonResponse(response.text())
      const parsed = JSON.parse(text)
      console.log("[LLMHelper] Parsed debug LLM response:", parsed)
      return parsed
    } catch (error) {
      console.error("Error debugging solution with images:", error)
      throw error
    }
  }

  private async fetchWeaviateContext(transcription: string): Promise<string> {
    try {
      const weaviateURL = process.env.WEAVIATE_URL
      const weaviateApiKey = process.env.WEAVIATE_API_KEY

      if (!weaviateURL || !weaviateApiKey) {
        console.warn('[LLMHelper] Weaviate credentials not configured, skipping context retrieval')
        return ''
      }

      const collectionName = process.env.WEAVIATE_COLLECTION || 'ProcurementContext'
      const contentField = process.env.WEAVIATE_CONTENT_FIELD || 'summary'

      // Use GraphQL API to query Weaviate with semantic search
      const response = await axios.post(
        `${weaviateURL}/v1/graphql`,
        {
          query: `
            {
              Get {
                ${collectionName}(
                  nearText: {
                    concepts: ["${transcription.replace(/"/g, '\\"')}"]
                  }
                  limit: 5
                ) {
                  ${contentField}
                  name
                  category
                  supplierId
                  annualSpend
                  _additional {
                    distance
                  }
                }
              }
            }
          `
        },
        {
          headers: {
            'Authorization': `Bearer ${weaviateApiKey}`,
            'Content-Type': 'application/json',
            'X-Weaviate-Cluster-Url': weaviateURL
          }
        }
      )

      console.log("[LLMHelper] Weaviate response:", JSON.stringify(response.data, null, 2))

      // Extract context from results
      let context = ''
      const results = response.data?.data?.Get?.[collectionName]
      if (results && results.length > 0) {
        console.log(`[LLMHelper] Found ${results.length} relevant context items from Weaviate`)
        context = results
          .map((obj: any, idx: number) => {
            const parts = []
            parts.push(`[Result ${idx + 1} - Distance: ${obj._additional?.distance?.toFixed(4)}]`)
            if (obj.name) parts.push(`Supplier: ${obj.name}`)
            if (obj.category) parts.push(`Category: ${obj.category}`)
            if (obj.supplierId) parts.push(`ID: ${obj.supplierId}`)
            if (obj.annualSpend) parts.push(`Annual Spend: $${obj.annualSpend}`)
            if (obj[contentField]) parts.push(`Details: ${obj[contentField]}`)
            return parts.join('\n')
          })
          .filter(Boolean)
          .join('\n\n')
      } else {
        console.log('[LLMHelper] No context found in Weaviate')
      }

      console.log("[LLMHelper] Retrieved context from Weaviate:", context)

      return context
    } catch (error) {
      console.error("[LLMHelper] Error fetching context from Weaviate:", error)
      if (error.response) {
        console.error("[LLMHelper] Weaviate error response:", JSON.stringify(error.response.data, null, 2))
      }
      return ''
    }
  }

  public async analyzeAudioFile(audioPath: string) {
    try {
      if (this.useOpenAI) {
        // Use OpenAI Whisper to transcribe audio
        console.log("[LLMHelper] Transcribing audio with Whisper API...")
        const transcription = await this.transcribeAudioWithWhisper(audioPath)
        console.log("[LLMHelper] Transcription:", transcription)
        
        // Fetch context from Weaviate
        console.log("[LLMHelper] Fetching context from Weaviate...")
        const context = await this.fetchWeaviateContext(transcription)
        
        // Build prompt with context if available
        let userPrompt = `The following is a transcription of an audio clip:\n\n"${transcription}"\n\n`
        
        if (context && context.trim().length > 0) {
          userPrompt += `\n**Relevant Context from Knowledge Base:**\n${context}\n\n`
          console.log("[LLMHelper] Using Weaviate context in prompt")
        } else {
          console.log("[LLMHelper] No Weaviate context available")
        }
        
        userPrompt += `Describe this content in a short, concise answer. In addition to your main answer, suggest several possible actions or responses the user could take next based on the content. Do not return a structured JSON object, just answer naturally as you would to a user.`
        
        const text = await this.callOpenAI([
          { role: 'system', content: this.systemPrompt },
          { role: 'user', content: userPrompt }
        ])
        
        return { text, timestamp: Date.now() }
      } else {
        // Gemini supports audio directly
        const audioData = await fs.promises.readFile(audioPath);
        const audioPart = {
          inlineData: {
            data: audioData.toString("base64"),
            mimeType: "audio/mp3"
          }
        };
        const prompt = `${this.systemPrompt}\n\nDescribe this audio clip in a short, concise answer. In addition to your main answer, suggest several possible actions or responses the user could take next based on the audio. Do not return a structured JSON object, just answer naturally as you would to a user.`;
        const result = await this.model.generateContent([prompt, audioPart]);
        const response = await result.response;
        const text = response.text();
        return { text, timestamp: Date.now() };
      }
    } catch (error) {
      console.error("Error analyzing audio file:", error);
      throw error;
    }
  }

  public async analyzeAudioFromBase64(data: string, mimeType: string) {
    try {
      if (this.useOpenAI) {
        // Use OpenAI Whisper to transcribe audio from base64
        console.log("[LLMHelper] Transcribing base64 audio with Whisper API...")
        const transcription = await this.transcribeAudioFromBase64WithWhisper(data, mimeType)
        console.log("[LLMHelper] Transcription:", transcription)
        
        // Fetch context from Weaviate
        console.log("[LLMHelper] Fetching context from Weaviate...")
        const context = await this.fetchWeaviateContext(transcription)
        
        // Build prompt with context if available
        let userPrompt = `The following is a transcription of an audio clip:\n\n"${transcription}"\n\n`
        
        if (context && context.trim().length > 0) {
          userPrompt += `\n**Relevant Context from Knowledge Base:**\n${context}\n\n`
          console.log("[LLMHelper] Using Weaviate context in prompt")
        } else {
          console.log("[LLMHelper] No Weaviate context available")
        }
        
        userPrompt += `Describe this content in a short, concise answer. In addition to your main answer, suggest several possible actions or responses the user could take next based on the content. Do not return a structured JSON object, just answer naturally as you would to a user and be concise.`
        
        const text = await this.callOpenAI([
          { role: 'system', content: this.systemPrompt },
          { role: 'user', content: userPrompt }
        ])
        
        return { text, timestamp: Date.now() }
      } else {
        // Gemini supports audio
        const audioPart = {
          inlineData: {
            data,
            mimeType
          }
        };
        const prompt = `${this.systemPrompt}\n\nDescribe this audio clip in a short, concise answer. In addition to your main answer, suggest several possible actions or responses the user could take next based on the audio. Do not return a structured JSON object, just answer naturally as you would to a user and be concise.`;
        const result = await this.model.generateContent([prompt, audioPart]);
        const response = await result.response;
        const text = response.text();
        return { text, timestamp: Date.now() };
      }
    } catch (error) {
      console.error("Error analyzing audio from base64:", error);
      throw error;
    }
  }

  public async analyzeImageFile(imagePath: string) {
    try {
      if (this.useOpenAI) {
        // OpenAI vision API
        const imageData = await fs.promises.readFile(imagePath);
        const base64Image = imageData.toString("base64");
        
        const text = await this.callOpenAI([
          {
            role: 'user',
            content: [
              { type: 'text', text: `${this.systemPrompt}\n\nDescribe the content of this image in a short, concise answer. In addition to your main answer, suggest several possible actions or responses the user could take next based on the image. Do not return a structured JSON object, just answer naturally as you would to a user. Be concise and brief.` },
              { type: 'image_url', image_url: { url: `data:image/png;base64,${base64Image}` } }
            ]
          }
        ])
        
        return { text, timestamp: Date.now() };
      } else {
        // Gemini or Ollama (Gemini only for now since Ollama doesn't support vision well)
        const imageData = await fs.promises.readFile(imagePath);
        const imagePart = {
          inlineData: {
            data: imageData.toString("base64"),
            mimeType: "image/png"
          }
        };
        const prompt = `${this.systemPrompt}\n\nDescribe the content of this image in a short, concise answer. In addition to your main answer, suggest several possible actions or responses the user could take next based on the image. Do not return a structured JSON object, just answer naturally as you would to a user. Be concise and brief.`;
        const result = await this.model.generateContent([prompt, imagePart]);
        const response = await result.response;
        const text = response.text();
        return { text, timestamp: Date.now() };
      }
    } catch (error) {
      console.error("Error analyzing image file:", error);
      throw error;
    }
  }

  public async chatWithGemini(message: string): Promise<string> {
    try {
      if (this.useOpenAI) {
        return this.callOpenAI([
          { role: 'system', content: this.systemPrompt },
          { role: 'user', content: message }
        ]);
      } else if (this.useOllama) {
        return this.callOllama(message);
      } else if (this.model) {
        const result = await this.model.generateContent(message);
        const response = await result.response;
        return response.text();
      } else {
        throw new Error("No LLM provider configured");
      }
    } catch (error) {
      console.error("[LLMHelper] Error in chatWithGemini:", error);
      throw error;
    }
  }

  public async chat(message: string): Promise<string> {
    return this.chatWithGemini(message);
  }

  public isUsingOllama(): boolean {
    return this.useOllama;
  }

  public isUsingOpenAI(): boolean {
    return this.useOpenAI;
  }

  public async getOllamaModels(): Promise<string[]> {
    if (!this.useOllama) return [];
    
    try {
      const response = await fetch(`${this.ollamaUrl}/api/tags`);
      if (!response.ok) throw new Error('Failed to fetch models');
      
      const data = await response.json();
      return data.models?.map((model: any) => model.name) || [];
    } catch (error) {
      console.error("[LLMHelper] Error fetching Ollama models:", error);
      return [];
    }
  }

  public getCurrentProvider(): "ollama" | "gemini" | "openai" {
    if (this.useOpenAI) return "openai";
    return this.useOllama ? "ollama" : "gemini";
  }

  public getCurrentModel(): string {
    if (this.useOpenAI) return this.openaiModel;
    return this.useOllama ? this.ollamaModel : "gemini-2.0-flash";
  }

  public async switchToOllama(model?: string, url?: string): Promise<void> {
    this.useOllama = true;
    if (url) this.ollamaUrl = url;
    
    if (model) {
      this.ollamaModel = model;
    } else {
      // Auto-detect first available model
      await this.initializeOllamaModel();
    }
    
    console.log(`[LLMHelper] Switched to Ollama: ${this.ollamaModel} at ${this.ollamaUrl}`);
  }

  public async switchToGemini(apiKey?: string): Promise<void> {
    if (apiKey) {
      const genAI = new GoogleGenerativeAI(apiKey);
      this.model = genAI.getGenerativeModel({ model: "gemini-2.0-flash" });
    }
    
    if (!this.model && !apiKey) {
      throw new Error("No Gemini API key provided and no existing model instance")
    }
    
    this.useOllama = false
    this.useOpenAI = false
    console.log("[LLMHelper] Switched to Gemini")
  }

  public async switchToOpenAI(apiKey: string, model?: string): Promise<void> {
    this.openaiApiKey = apiKey
    this.openaiModel = model || "gpt-4o"
    this.useOpenAI = true
    this.useOllama = false
    console.log(`[LLMHelper] Switched to OpenAI: ${this.openaiModel}`)
  }

  public async testConnection(): Promise<{ success: boolean; error?: string }> {
    try {
      if (this.useOpenAI) {
        // Test OpenAI with a simple prompt
        const result = await this.callOpenAI([
          { role: 'user', content: 'Hello' }
        ]);
        if (result) {
          return { success: true };
        } else {
          return { success: false, error: "Empty response from OpenAI" };
        }
      } else if (this.useOllama) {
        const available = await this.checkOllamaAvailable();
        if (!available) {
          return { success: false, error: `Ollama not available at ${this.ollamaUrl}` };
        }
        // Test with a simple prompt
        await this.callOllama("Hello");
        return { success: true };
      } else {
        if (!this.model) {
          return { success: false, error: "No Gemini model configured" };
        }
        // Test with a simple prompt
        const result = await this.model.generateContent("Hello");
        const response = await result.response;
        const text = response.text(); // Ensure the response is valid
        if (text) {
          return { success: true };
        } else {
          return { success: false, error: "Empty response from Gemini" };
        }
      }
    } catch (error) {
      return { success: false, error: error.message };
    }
  }
}