#!/usr/bin/env python3
"""
Pre-Deployment Verification Script
Run this before deploying to 360 F1 server to ensure everything is ready
"""

import os
import sys
import sqlite3
from pathlib import Path

def check_file(filepath, required=True):
    """Check if a file exists"""
    exists = os.path.exists(filepath)
    status = "✅" if exists else ("❌" if required else "⚠️")
    print(f"{status} {filepath}")
    return exists

def check_folder(folderpath, required=True):
    """Check if a folder exists and has files"""
    exists = os.path.exists(folderpath)
    if exists and os.path.isdir(folderpath):
        file_count = len(os.listdir(folderpath))
        print(f"✅ {folderpath} ({file_count} files)")
        return True
    else:
        status = "❌" if required else "⚠️"
        print(f"{status} {folderpath}")
        return False

def check_database():
    """Check database tables"""
    if not os.path.exists('combined_db.db'):
        print("⚠️  Database doesn't exist yet (will be created on first run)")
        return True
    
    try:
        conn = sqlite3.connect('combined_db.db')
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        required_tables = [
            'evaluations', 'qa_history', 'qa_feedback', 
            'feedback', 'handbook_feedback', 'interview_questions', 
            'recruiter_handbooks'
        ]
        
        print("\n📊 Database Tables:")
        all_present = True
        for table in required_tables:
            if table in tables:
                print(f"  ✅ {table}")
            else:
                print(f"  ❌ {table}")
                all_present = False
        
        conn.close()
        return all_present
    except Exception as e:
        print(f"⚠️  Error checking database: {e}")
        return False

def check_env_template():
    """Check if .env exists and has required keys"""
    if not os.path.exists('.env'):
        print("\n⚠️  .env file not found")
        print("   You'll need to create this on the server with:")
        print("   - GROQ_API_KEY")
        print("   - PINECONE_API_KEY")
        print("   - GEMINI_API_KEY")
        return False
    
    print("\n✅ .env file exists")
    with open('.env', 'r') as f:
        content = f.read()
        required_keys = ['GROQ_API_KEY', 'PINECONE_API_KEY', 'GEMINI_API_KEY']
        for key in required_keys:
            if key in content:
                print(f"  ✅ {key} found")
            else:
                print(f"  ⚠️  {key} not found")
    return True

def main():
    print("=" * 60)
    print("🔍 PRE-DEPLOYMENT VERIFICATION")
    print("=" * 60)
    
    print("\n📁 Essential Files:")
    check_file("app.py", required=True)
    check_file("run_production.py", required=True)
    check_file("requirements.txt", required=True)
    check_file(".gitignore", required=True)
    
    print("\n📂 Essential Folders:")
    check_folder("templates", required=True)
    check_folder("static", required=True)
    hr_docs_ok = check_folder("HR_docs", required=True)
    check_folder("uploads", required=False)
    
    print("\n📄 Key Template Files:")
    check_file("templates/base.html", required=True)
    check_file("templates/base2.html", required=True)
    check_file("templates/index2.html", required=True)
    check_file("templates/feedback_history.html", required=True)
    
    print("\n🎨 Static Assets:")
    check_file("static/js/resume-evaluator.js", required=True)
    check_file("static/css/style.css", required=True)
    
    # Check database
    check_database()
    
    # Check environment
    check_env_template()
    
    print("\n" + "=" * 60)
    
    # Final summary
    issues = []
    
    if not os.path.exists("run_production.py"):
        issues.append("Missing run_production.py")
    
    if not hr_docs_ok:
        issues.append("HR_docs folder missing or empty")
    
    if not os.path.exists("requirements.txt"):
        issues.append("Missing requirements.txt")
    
    if issues:
        print("❌ ISSUES FOUND:")
        for issue in issues:
            print(f"   - {issue}")
        print("\n⚠️  Please fix these issues before deploying")
        return 1
    else:
        print("✅ ALL CHECKS PASSED!")
        print("\n🚀 Ready to deploy to 360 F1 server!")
        print("\nNext steps:")
        print("1. Push to git repository")
        print("2. Clone on server")
        print("3. Create .env with API keys")
        print("4. Install dependencies: pip install -r requirements.txt")
        print("5. Run: python run_production.py")
        return 0

if __name__ == "__main__":
    sys.exit(main())

