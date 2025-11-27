#!/bin/bash

echo "=========================================="
echo "  🚀 GitHub Repository Setup"
echo "=========================================="
echo ""

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "📦 Initializing git repository..."
    git init
    echo "✅ Git initialized"
else
    echo "✅ Git repository already initialized"
fi

# Add all files
echo ""
echo "📁 Adding files to git..."
git add .

# Create initial commit
echo ""
echo "💾 Creating initial commit..."
git commit -m "Initial commit: ML Pipeline with CI/CD" || echo "⚠️  Nothing to commit or already committed"

# Set main branch
echo ""
echo "🌿 Setting main branch..."
git branch -M main

# Instructions for adding remote
echo ""
echo "=========================================="
echo "  📝 Next Steps"
echo "=========================================="
echo ""
echo "1. Create a new repository on GitHub:"
echo "   https://github.com/new"
echo ""
echo "2. Run these commands (replace with your repo URL):"
echo ""
echo "   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git"
echo "   git push -u origin main"
echo ""
echo "3. Go to GitHub Actions tab to see your CI/CD pipeline!"
echo ""
echo "=========================================="
echo "  ✅ Git setup complete!"
echo "=========================================="
