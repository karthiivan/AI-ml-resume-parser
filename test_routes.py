#!/usr/bin/env python3
"""
Quick test to verify all routes are working
"""

import requests
import sys

def test_routes():
    """Test main application routes"""
    base_url = "http://127.0.0.1:5000"
    
    routes_to_test = [
        "/",
        "/about", 
        "/features",
        "/contact",
        "/auth/login",
        "/auth/register"
    ]
    
    print("🧪 Testing Application Routes")
    print("-" * 40)
    
    all_passed = True
    
    for route in routes_to_test:
        try:
            response = requests.get(f"{base_url}{route}", timeout=5)
            if response.status_code == 200:
                print(f"✅ {route} - OK")
            else:
                print(f"❌ {route} - Status: {response.status_code}")
                all_passed = False
        except requests.exceptions.RequestException as e:
            print(f"❌ {route} - Error: {e}")
            all_passed = False
    
    print("-" * 40)
    if all_passed:
        print("🎉 All routes are working!")
        return True
    else:
        print("⚠️ Some routes have issues")
        return False

if __name__ == "__main__":
    success = test_routes()
    sys.exit(0 if success else 1)
