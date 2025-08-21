# README.md
# Cognitive Architecture Orchestrator

An enterprise-grade AI agent management system inspired by neuroscience and cognitive architectures. This system implements a multi-agent framework with conscious/unconscious processing, memory management, and self-learning capabilities.

## 🧠 Architecture Overview

The system consists of 8 core components:

1. **Receiver** - Input gateway and event normalization
2. **Self-Agent** - Core learning and exploration orchestrator  
3. **Conscious Agent** - Deliberative reasoning and planning (System 2)
4. **Unconscious Agent** - Background processing and reflexes (System 1)
5. **Relationship Manager** - Context and social interaction management
6. **Memory Cache** - Short-term working memory
7. **Retriever** - Long-term knowledge storage and retrieval
8. **Flywheel** - Meta-learning and system evolution

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- Azure OpenAI API access
- Windows/Linux/Mac

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd cognitive-architecture
   ```

2. **Set up environment**
   
   **Windows:**
   ```bash
   run.bat
   ```
   
   **Linux/Mac:**
   ```bash
   chmod +x run.sh
   ./run.sh
   ```

3. **Configure environment**
   - Copy `.env.example` to `.env`
   - Configure your Azure OpenAI settings:
   ```
   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
   AZURE_OPENAI_API_KEY=your-api-key-here
   AZURE_OPENAI_DEPLOYMENT=gpt-4
   ```

4. **Access the application**
   - Dashboard: http://localhost:8080/
   - Agent Management: http://localhost:8080/agents
   - Monitoring: http://localhost:8080/monitoring
   - Knowledge Base: http://localhost:8080/knowledge

## 📋 Features

### 🎛️ Agent Management
- Real-time agent status monitoring
- Start/stop/restart individual agents
- Configuration management
- Performance metrics tracking

### 📊 System Monitoring
- Live performance dashboards
- System health indicators
- Resource usage tracking
- Real-time logging

### 🧠 Knowledge Management
- Hybrid search (semantic + keyword)
- Document upload and processing
- Concept graph visualization
- Knowledge analytics

### 🔄 Self-Learning
- Automatic system optimization
- A/B testing framework
- Concept evolution and merging
- Policy adaptation

## 🏗️ Project Structure

```
cognitive-architecture/
├── app.py                 # Main FastAPI application
├── startup.py            # System initialization
├── requirements.txt      # Python dependencies
├── .env.example         # Environment template
├── run.bat / run.sh     # Startup scripts
├── core/
│   ├── config.py        # Configuration management
│   ├── database.py      # Database models and operations
│   ├── logger.py        # Logging setup
│   └── llm_client.py    # Azure OpenAI client
├── components/
│   ├── receiver.py      # Input gateway
│   ├── self_agent.py    # Core learning agent
│   ├── conscious_agent.py    # Deliberative reasoning
│   ├── unconscious_agent.py  # Background processing
│   ├── relationship_manager.py  # Context management
│   ├── memory_cache.py       # Working memory
│   ├── retriever.py          # Knowledge retrieval
│   └── flywheel.py          # Meta-learning
├── api/
│   ├── orchestrator.py  # System orchestration endpoints
│   ├── agents.py        # Agent management API
│   └── monitoring.py    # Monitoring and metrics API
├── templates/
│   └── pages/
│       ├── dashboard.html   # Main dashboard
│       ├── agents.html      # Agent management
│       ├── monitoring.html  # System monitoring
│       └── knowledge.html   # Knowledge management
└── static/
    └── style.css        # Additional styles
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL | `https://your-resource.openai.azure.com/` |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key | `your-api-key-here` |
| `AZURE_OPENAI_DEPLOYMENT` | Model deployment name | `gpt-4` |
| `APP_ENV` | Application environment | `dev` or `prod` |
| `HOST` | Server host | `0.0.0.0` |
| `PORT` | Server port | `8080` |
| `LOG_LEVEL` | Logging level | `INFO` |

### Agent Configuration

Each agent can be configured through the web interface or API:

- **Processing intervals**
- **Memory limits**  
- **Learning parameters**
- **Routing rules**
- **Performance thresholds**

## 📖 API Documentation

### Orchestrator Endpoints

- `GET /api/orchestrator/status` - System status
- `POST /api/orchestrator/test-event` - Process test event
- `POST /api/orchestrator/start-agent/{name}` - Start agent
- `POST /api/orchestrator/stop-agent/{name}` - Stop agent

### Agent Management

- `GET /api/agents/status` - All agent status
- `GET /api/agents/{name}/status` - Specific agent status  
- `PUT /api/agents/{name}/config` - Update agent configuration

### Monitoring

- `GET /api/monitoring/metrics` - Performance metrics
- `GET /api/monitoring/logs` - System logs
- `GET /api/monitoring/health-summary` - Health summary

## 🧪 Testing

Run system tests:
```bash
python -m pytest tests/
```

Run individual component tests:
```bash
python -m pytest tests/test_agents.py
```

## 🔍 Troubleshooting

### Common Issues

1. **Azure OpenAI Connection Failed**
   - Verify API key and endpoint in `.env`
   - Check API quota and billing status
   - Ensure correct model deployment name

2. **Database Errors**
   - Check write permissions in application directory
   - Verify SQLite installation
   - Clear database file if corrupted: `rm cognitive_architecture.db`

3. **Agent Not Starting**
   - Check agent logs in monitoring panel
   - Verify configuration parameters
   - Restart system if needed

4. **Memory Issues**
   - Monitor system resources in monitoring panel
   - Adjust agent memory limits
   - Clear caches through web interface

### Debug Mode

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
python startup.py
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -am 'Add new feature'`
4. Push to branch: `git push origin feature/new-feature`
5. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Inspired by cognitive architectures and neuroscience research
- Built with FastAPI, Azure OpenAI, and modern web technologies
- Enterprise design patterns and best practices

---

For support and questions, please open an issue or contact the development team.