# 🛡️ SQL Agent Security & Analytics Masterclass

A comprehensive educational repository demonstrating the progression from basic SQL agents to production-ready, secure analytics systems using LangChain and OpenAI. Learn by example how to build safe, powerful SQL agents for business intelligence.

## 🎯 What You'll Learn

This repository teaches you to build SQL agents with **progressive security implementations**, from completely unrestricted (dangerous) to production-ready (secure). Each script builds upon the previous one, showing the evolution of security practices.

### 📚 Learning Progression

0. **Simple LLM** → Pure language model usage without agents or tools
1. **Basic SQL Agent** → Simple database querying with built-in tools
2. **Dangerous Agent** → What NOT to do (security vulnerabilities)
3. **Secure Agent** → Production-ready guardrails and validation
4. **Analytics Agent** → Advanced business intelligence capabilities

### 🔒 Security Concepts Covered

- **Input validation** and SQL injection prevention
- **Whitelist-based security** (only allow SELECT statements)
- **Result set limiting** to prevent resource exhaustion
- **Multi-statement prevention** to block SQL injection attacks
- **Error handling** and graceful failure modes
- **Schema-based restrictions** for table access control

### 📊 Analytics Features Demonstrated

- **Complex multi-table JOINs** for business intelligence
- **Revenue analysis** and customer lifetime value calculations
- **Time-series analysis** with date functions and aggregations
- **Product performance rankings** and trend analysis
- **Multi-turn conversations** for iterative data exploration

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key
- Basic understanding of SQL and Python

### Setup Instructions

```bash
# 1. Create and activate virtual environment (from project root)
python -m venv .venv && source .venv/bin/activate      # or .venv\Scripts\activate on Windows

# 2. Install dependencies (from project root)
pip install -r requirements.txt

# 3. Configure environment variables (from project root, if .env doesn't exist)
cp .env.example .env
# Edit .env and add your OpenAI API key: OPENAI_API_KEY=sk-...

# 4. Navigate to the SQLAgent folder
cd SQLAgent

# 5. (Optional) Reset database to initial state
python scripts/reset_db.py

# 6. Run the tutorial scripts in order
python scripts/01_simple_agent.py       # Basic SQL agent
python scripts/02_risky_delete_demo.py  # ⚠️ Dangerous patterns (educational)
python scripts/03_guardrailed_agent.py  # Secure implementation
python scripts/04_complex_queries.py    # Advanced analytics
```

## 📁 Repository Structure

```
Project Root/
├── ⚙️ requirements.txt                 # Python dependencies (shared by all projects)
├── 🔐 .env.example                     # Environment variables template
├── 🔐 .env                             # Your environment variables (with API key)
├── 🗂️ .venv/                           # Virtual environment (shared)
└── 📂 SQLAgent/                        # SQL Agent Educational Package
    ├── 📊 sql_agent_class.db           # Pre-built SQLite database with sample data
    ├── 🔄 sql_agent_seed.sql           # Database schema and seed data (idempotent)
    ├── 📖 README.md                     # This comprehensive guide
    └── 📂 scripts/                      # Progressive tutorial scripts
        ├── 🔄 reset_db.py               # Database reset utility
        ├── 0️⃣ 00_simple_llm.py          # Simple LLM usage (no agents/tools)
        ├── 1️⃣ 01_simple_agent.py        # Basic SQL agent implementation
        ├── ⚠️ 02_risky_delete_demo.py    # Dangerous patterns (educational only)
        ├── 🛡️ 03_guardrailed_agent.py   # Secure SQL agent with guardrails
        └── 📈 04_complex_queries.py      # Advanced analytics capabilities
```

## 🔍 Detailed Script Explanations

### 1️⃣ `01_simple_agent.py` - Basic SQL Agent

**Purpose**: Introduction to LangChain SQL agents
**Security Level**: ❌ None (unrestricted access)
**Use Case**: Learning basic agent concepts

**Key Features**:
- Uses LangChain's built-in `create_sql_agent()`
- No security restrictions whatsoever
- Can execute ANY SQL including DELETE, DROP, etc.
- Simple demonstration of agent capabilities

**⚠️ Security Issues**:
- No input validation
- No operation restrictions
- No result set limits
- Not suitable for any production use

### 2️⃣ `02_risky_delete_demo.py` - Dangerous Agent Demo

**Purpose**: Educational example of what NOT to do
**Security Level**: ❌ Actively dangerous
**Use Case**: Understanding security vulnerabilities

**Key Features**:
- Custom tool that executes ANY SQL
- Explicitly allows destructive operations
- Demonstrates actual data deletion
- Shows transaction commit behavior

**⚠️ Security Issues** (intentional for education):
- Zero input validation or sanitization
- Allows DELETE, DROP, TRUNCATE operations
- No user permissions or access controls
- Direct SQL execution without safety checks
- Automatic transaction commits make changes permanent

**Educational Value**:
- Shows exactly why guardrails are needed
- Demonstrates real consequences of unrestricted access
- Provides basis for understanding security improvements

### 3️⃣ `03_guardrailed_agent.py` - Secure Implementation

**Purpose**: Production-ready secure SQL agent
**Security Level**: ✅ High (multiple guardrails)
**Use Case**: Safe analytics and reporting

**Security Features**:
- ✅ **Input validation** using regex patterns
- ✅ **Whitelist approach** - only SELECT statements allowed
- ✅ **Automatic LIMIT injection** to prevent large result sets
- ✅ **SQL injection protection** through pattern matching
- ✅ **Multiple statement prevention** to block chained attacks
- ✅ **Comprehensive error handling** with informative messages
- ✅ **Read-only operations** only - no data modification possible

**Technical Implementation**:
- Custom `SafeSQLTool` class with validation layers
- Regex-based dangerous operation detection
- Performance optimization through result limiting
- Structured error handling and reporting

### 4️⃣ `04_complex_queries.py` - Advanced Analytics

**Purpose**: Business intelligence and complex analytics
**Security Level**: ✅ High (inherits all guardrails from script 03)
**Use Case**: Production analytics and reporting systems

**Advanced Features**:
- 📊 **Complex multi-table JOINs** for comprehensive analysis
- 📈 **Revenue analysis** with sophisticated business logic
- 🕐 **Time-series analysis** for trend identification
- 👥 **Customer segmentation** and lifetime value calculations
- 🏆 **Performance rankings** and comparative analysis
- 💬 **Multi-turn conversations** for iterative data exploration

**Analytics Capabilities Demonstrated**:
- Product revenue analysis with cross-table aggregations
- Weekly trend analysis using date functions
- Customer lifecycle analysis with segmentation
- Lifetime value rankings with complex calculations
- Category performance analysis with drill-down capabilities

## 🗄️ Database Schema

The included SQLite database contains realistic e-commerce data:

### Core Tables
- **`customers`** - Customer information (id, name, email, created_at, region)
- **`orders`** - Order records (id, customer_id, order_date, status)
- **`order_items`** - Line items (id, order_id, product_id, quantity, unit_price_cents)
- **`products`** - Product catalog (id, name, category, description)
- **`payments`** - Payment records (id, order_id, amount_cents, method)
- **`refunds`** - Refund tracking (id, order_id, amount_cents, reason)

### Sample Business Metrics
- **Revenue calculation**: `sum(quantity * unit_price_cents) - refunds.amount_cents`
- **Customer lifetime value**: Total revenue per customer minus refunds
- **Product performance**: Revenue and units sold by product/category
- **Time-series analysis**: Weekly/monthly revenue trends

## 🛡️ Security Best Practices Demonstrated

### 1. Input Validation
```python
# Multi-layer validation approach
if re.search(r"\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|CREATE|REPLACE)\b", s, re.I):
    return "ERROR: write operations are not allowed."

if ";" in s:
    return "ERROR: multiple statements are not allowed."

if not re.match(r"(?is)^\s*select\b", s):
    return "ERROR: only SELECT statements are allowed."
```

### 2. Result Set Limiting
```python
# Automatic LIMIT injection for performance
if not re.search(r"\blimit\s+\d+\b", s, re.I):
    s += " LIMIT 200"  # Conservative limit
```

### 3. Error Handling
```python
# Comprehensive error catching and reporting
try:
    result = conn.exec_driver_sql(s)
    # ... process results
except Exception as e:
    return f"ERROR: {e}"  # Safe error reporting
```

## 🎓 Educational Workflow

### Recommended Learning Path

1. **Start with `01_simple_agent.py`**
   - Understand basic LangChain SQL agent concepts
   - See how easy it is to create an unrestricted agent
   - Note the security implications

2. **Run `02_risky_delete_demo.py`** (carefully!)
   - Observe how dangerous unrestricted access can be
   - See actual data deletion in action
   - Understand why security measures are essential

3. **Study `03_guardrailed_agent.py`**
   - Learn comprehensive security implementation
   - Understand each validation layer
   - See how to maintain functionality while adding security

4. **Explore `04_complex_queries.py`**
   - Discover advanced analytics capabilities
   - Learn business intelligence patterns
   - Practice with multi-turn conversations

### Key Learning Objectives

- **Security mindset**: Always validate and restrict agent capabilities
- **Progressive enhancement**: Build security incrementally
- **Error handling**: Provide helpful feedback without exposing vulnerabilities
- **Performance consideration**: Limit result sets and prevent resource exhaustion
- **Business context**: Understand the domain for better analytics

## ⚠️ Safety Guidelines

### Development Environment
- ✅ Use the provided SQLite database for learning
- ✅ Test all security features thoroughly
- ✅ Understand each validation layer before proceeding

### Production Considerations
- ❌ **NEVER** use the dangerous patterns from script 02 in production
- ✅ Always use read-only database users for analytics agents
- ✅ Implement comprehensive logging and monitoring
- ✅ Regular security audits and testing
- ✅ Schema-based access controls and table restrictions

### Risk Mitigation
- **Database isolation**: Use dedicated analytics databases
- **User permissions**: Implement least-privilege access
- **Monitoring**: Log all agent queries and results
- **Rate limiting**: Prevent resource exhaustion attacks
- **Regular audits**: Review agent behavior and query patterns

## 🔧 Troubleshooting

### Common Issues

**"ModuleNotFoundError" for LangChain packages**
```bash
# Ensure virtual environment is activated and dependencies installed
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

**"OpenAI API key not found"**
```bash
# Verify .env file configuration
cp .env.example .env
# Edit .env and add: OPENAI_API_KEY=sk-your-key-here
```

**"Database file not found"**
```bash
# Reset the database
python scripts/reset_db.py
```

**Pydantic validation errors**
- Ensure you're using Python 3.8+ with compatible package versions
- Check that type annotations are properly formatted

### Getting Help

- Check the comprehensive inline documentation in each script
- Review error messages carefully - they provide specific guidance
- Test with simple queries first before attempting complex analytics
- Use the Jupyter notebook for interactive exploration

## 🤝 Contributing

This is an educational repository. Contributions that enhance the learning experience are welcome:

- **Documentation improvements**: Clearer explanations or additional examples
- **Security enhancements**: Additional validation patterns or safety measures
- **Analytics examples**: More sophisticated business intelligence queries
- **Error handling**: Better user experience and debugging information

## 📄 License

Educational use encouraged. Please reference this repository when using the patterns in your own projects.

---

**🎯 Remember**: The goal is to understand the progression from dangerous to secure SQL agent implementations. Start with the basics, understand the risks, then build robust, production-ready solutions.