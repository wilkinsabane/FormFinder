# Football Data Visualization Dashboard

A comprehensive Streamlit-based web dashboard for visualizing football data stored in PostgreSQL database.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PostgreSQL database with football data
- Existing FormFinder project setup

### Installation

1. **Install additional dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure database connection**:
   ```bash
   # Copy the secrets template
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   
   # Edit with your database credentials
   # For Windows: notepad .streamlit/secrets.toml
   # For Mac/Linux: nano .streamlit/secrets.toml
   ```

3. **Set environment variables (alternative)**:
   ```bash
   # Windows PowerShell
   $env:DB_URI="postgresql://username:password@localhost:5432/formfinder"
   
   # Linux/Mac
   export DB_URI="postgresql://username:password@localhost:5432/formfinder"
   ```

### Running the Dashboard

#### Local Development
```bash
# Run the dashboard locally
streamlit run dashboard.py

# Or use the convenience script
python -m streamlit run dashboard.py
```

#### Production Deployment
```bash
# For production with optimized settings
streamlit run dashboard.py --server.port=8501 --server.address=0.0.0.0
```

## 📊 Features

### Interactive Visualizations
- **Teams**: Bar charts showing team performance metrics
- **Standings**: League standings with trend analysis
- **Predictions**: Match predictions with confidence scores
- **Leagues**: Distribution charts across different leagues
- **High Form Teams**: Highlighted teams with exceptional performance
- **Fixtures**: Timeline and calendar views of upcoming matches
- **Data Logs**: Success rates and fetch timeline visualization

### Filtering & Controls
- **League Selection**: Filter data by specific leagues
- **Date Range**: Dynamic date filtering for fixtures and logs
- **Real-time Refresh**: Manual data refresh with caching
- **Responsive Design**: Optimized for desktop and mobile viewing

### Performance Features
- **Caching**: 5-minute cache for improved performance
- **Lazy Loading**: Data fetched only when needed
- **Pagination**: Automatic pagination for large datasets
- **Error Handling**: Graceful error handling with user-friendly messages

## 🔧 Configuration

### Database Schema Requirements
The dashboard expects the following tables in your PostgreSQL database:
- `teams`: Team information and statistics
- `standings`: League standings data
- `predictions`: Match predictions and analysis
- `leagues`: League information
- `high_form_teams`: Teams with exceptional recent performance
- `fixtures`: Upcoming and past match fixtures
- `data_fetch_logs`: Data fetching operation logs

### Customization

#### Styling
Edit `.streamlit/config.toml` to customize:
- Theme colors
- Font preferences
- Server settings

#### Adding New Visualizations
1. Add new query methods in the `DatabaseManager` class
2. Create new tab rendering methods in the `Dashboard` class
3. Follow the existing pattern for consistency

#### Performance Tuning
- Adjust cache TTL in `@st.cache_data(ttl=300)` decorators
- Modify query limits for large datasets
- Add database indexes for frequently queried columns

## 🐛 Troubleshooting

### Common Issues

**Database Connection Error**:
```
Error: Database connection failed
```
- Verify PostgreSQL is running
- Check database credentials in secrets.toml
- Ensure database exists and is accessible

**Missing Tables**:
```
Error: relation "teams" does not exist
```
- Run database setup scripts from the main project
- Verify schema migration completed successfully

**Port Already in Use**:
```
Error: Address already in use
```
- Change port in config.toml: `port = 8502`
- Or kill existing process: `lsof -ti:8501 | xargs kill -9`

### Debug Mode
Enable debug logging:
```bash
streamlit run dashboard.py --logger.level=debug
```

## 🚀 Deployment Options

### Streamlit Cloud (Recommended)
1. Push code to GitHub repository
2. Connect repository to Streamlit Cloud
3. Set environment variables in Streamlit Cloud dashboard

### Docker Deployment
```bash
# Build Docker image
docker build -t football-dashboard .

# Run container
docker run -p 8501:8501 -e DB_URI=postgresql://... football-dashboard
```

### Heroku Deployment
```bash
# Install Heroku CLI and login
heroku create football-dashboard-app
git push heroku main
heroku config:set DB_URI=postgresql://...
```

### AWS EC2 Deployment
```bash
# SSH into EC2 instance
ssh -i your-key.pem ec2-user@your-instance

# Install dependencies
pip install -r requirements.txt

# Run with systemd
sudo systemctl enable streamlit-dashboard
sudo systemctl start streamlit-dashboard
```

## 📈 Usage Analytics

### Monitoring
The dashboard includes built-in logging:
- Database query performance
- User interaction tracking
- Error monitoring

### Custom Metrics
Add custom analytics by modifying the `render_sidebar_stats()` method:
```python
# Example: Add custom metric
custom_metric = self.db.fetch_data("SELECT COUNT(*) FROM custom_table")
st.sidebar.metric("Custom Metric", custom_metric.iloc[0]['count'])
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-visualization`
3. Commit changes: `git commit -am 'Add new visualization'`
4. Push to branch: `git push origin feature/new-visualization`
5. Submit pull request

## 📄 License

This project is part of the FormFinder ecosystem. See main project LICENSE file.

## 📞 Support

For issues and questions:
1. Check this README and main project documentation
2. Review existing GitHub issues
3. Create new issue with detailed description
4. Include error logs and system information

## 🔗 Related Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Python](https://plotly.com/python/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [FormFinder Main README](README.md)
- [Project Architecture](ARCHITECTURE.md)