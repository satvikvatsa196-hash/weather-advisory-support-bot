import { useState, useRef, useEffect } from 'react'
import { sendMessage } from './api'

function App() {
  const [messages, setMessages] = useState<{role: 'user' | 'bot', content: string}[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  
  // Initialize session ID once per page load
  const sessionIdRef = useRef<string>(crypto.randomUUID())
  const scrollRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages, isLoading, error])

  const handleSend = async () => {
    if (!input.trim() || isLoading) return
    
    const userMessage = input.trim()
    setInput('')
    setError(null)
    setMessages(prev => [...prev, { role: 'user', content: userMessage }])
    setIsLoading(true)
    
    try {
      const result = await sendMessage(sessionIdRef.current, userMessage)
      setMessages(prev => [...prev, { role: 'bot', content: result.response }])
    } catch (err) {
      console.error(err)
      setError("Unable to reach the weather advisory service.")
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="chat-container">
      <header>
        <h1>Weather Advisory</h1>
      </header>
      
      <main className="message-area" ref={scrollRef}>
        {messages.length === 0 && (
          <div className="message bot" style={{ alignSelf: 'center', opacity: 0.7, background: 'none' }}>
            Ask about outdoor activity safety to get started...
          </div>
        )}
        {messages.map((msg, idx) => (
          <div key={idx} className={`message ${msg.role}`}>
            <strong>{msg.role === 'user' ? 'You' : 'Bot'}:</strong> 
            <span style={{ whiteSpace: 'pre-wrap', display: 'block', marginTop: '0.25rem' }}>{msg.content}</span>
          </div>
        ))}
        {isLoading && (
          <div className="message bot" style={{ fontStyle: 'italic', opacity: 0.7 }}>
            Thinking...
          </div>
        )}
        {error && (
          <div className="message bot" style={{ color: 'red', background: '#ffebee' }}>
            {error}
          </div>
        )}
      </main>

      <footer className="input-area">
        <input 
          type="text" 
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSend()}
          placeholder="Ask about outdoor activity safety..."
          disabled={isLoading}
        />
        <button onClick={handleSend} disabled={isLoading || !input.trim()}>
          {isLoading ? 'Sending...' : 'Send'}
        </button>
      </footer>
    </div>
  )
}

export default App
