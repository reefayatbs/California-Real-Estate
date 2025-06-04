#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}California Real Estate Price Predictor Setup${NC}\n"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 is not installed. Please install Python 3 and try again.${NC}"
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo -e "${RED}Node.js is not installed. Please install Node.js and try again.${NC}"
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo -e "${RED}npm is not installed. Please install npm and try again.${NC}"
    exit 1
fi

# Create Python virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo -e "${BLUE}Creating Python virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
source venv/bin/activate

# Install Python dependencies
echo -e "${BLUE}Installing Python dependencies...${NC}"
pip install -r requirements.txt

# Install frontend dependencies
echo -e "${BLUE}Installing frontend dependencies...${NC}"
cd client
npm install

# Start backend server in background
echo -e "${BLUE}Starting backend server...${NC}"
cd ../server
python server.py & 
BACKEND_PID=$!

# Start frontend
echo -e "${BLUE}Starting frontend...${NC}"
cd ../client
npm start

# Cleanup function
cleanup() {
    echo -e "\n${BLUE}Shutting down servers...${NC}"
    kill $BACKEND_PID
    exit 0
}

# Set up cleanup on script termination
trap cleanup SIGINT SIGTERM

# Wait for processes
wait 