#!/bin/bash

# Advanced Fraud Detection System Deployment Script

set -e  # Exit on any error

echo "🚀 Starting Advanced Fraud Detection System Deployment"
echo "======================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    print_status "Checking Docker installation..."
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    print_status "Docker and Docker Compose are installed ✅"
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    mkdir -p models data logs explainability_results
    print_status "Directories created ✅"
}

# Build Docker image
build_image() {
    print_status "Building Docker image..."
    docker build -t fraud-detection-api:latest .
    
    if [ $? -eq 0 ]; then
        print_status "Docker image built successfully ✅"
    else
        print_error "Failed to build Docker image"
        exit 1
    fi
}

# Start services
start_services() {
    print_status "Starting services with Docker Compose..."
    docker-compose up -d fraud-detection-api
    
    if [ $? -eq 0 ]; then
        print_status "Services started successfully ✅"
    else
        print_error "Failed to start services"
        exit 1
    fi
}

# Wait for API to be ready
wait_for_api() {
    print_status "Waiting for API to be ready..."
    max_attempts=30
    attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s http://localhost:5000/health > /dev/null 2>&1; then
            print_status "API is ready! ✅"
            return 0
        fi
        
        echo "Attempt $attempt/$max_attempts - API not ready yet..."
        sleep 2
        ((attempt++))
    done
    
    print_error "API failed to start within expected time"
    return 1
}

# Run tests
run_tests() {
    print_status "Running API tests..."
    
    if command -v python3 &> /dev/null; then
        # Install required packages for testing
        pip3 install requests > /dev/null 2>&1 || print_warning "Could not install requests package"
        
        python3 test_api.py
        
        if [ $? -eq 0 ]; then
            print_status "All tests passed! ✅"
        else
            print_warning "Some tests failed. Check the output above."
        fi
    else
        print_warning "Python3 not found. Skipping automated tests."
        print_status "You can manually test the API at http://localhost:5000"
    fi
}

# Show deployment information
show_info() {
    echo ""
    echo "🎉 Deployment Complete!"
    echo "======================"
    echo ""
    echo "📡 API Endpoints:"
    echo "  • Main API: http://localhost:5000"
    echo "  • Health Check: http://localhost:5000/health"
    echo "  • API Documentation: http://localhost:5000"
    echo ""
    echo "🔧 Available Services:"
    echo "  • Fraud Detection: POST /predict/fraud"
    echo "  • Credit Card Fraud: POST /predict/credit"
    echo "  • Explanations: POST /explain/fraud, POST /explain/credit"
    echo "  • Batch Processing: POST /batch/fraud, POST /batch/credit"
    echo ""
    echo "📊 Management Commands:"
    echo "  • View logs: docker-compose logs -f fraud-detection-api"
    echo "  • Stop services: docker-compose down"
    echo "  • Restart: docker-compose restart"
    echo "  • Update: ./deploy.sh"
    echo ""
    echo "🧪 Test the API:"
    echo "  • Run tests: python3 test_api.py"
    echo "  • Manual test: curl http://localhost:5000/health"
    echo ""
}

# Cleanup function
cleanup() {
    print_status "Cleaning up old containers and images..."
    docker-compose down > /dev/null 2>&1 || true
    docker system prune -f > /dev/null 2>&1 || true
}

# Main deployment process
main() {
    echo "Starting deployment process..."
    
    # Parse command line arguments
    SKIP_TESTS=false
    CLEANUP=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            --cleanup)
                CLEANUP=true
                shift
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --skip-tests    Skip running API tests"
                echo "  --cleanup       Clean up old containers before deployment"
                echo "  --help          Show this help message"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
    
    # Cleanup if requested
    if [ "$CLEANUP" = true ]; then
        cleanup
    fi
    
    # Run deployment steps
    check_docker
    create_directories
    build_image
    start_services
    
    if wait_for_api; then
        if [ "$SKIP_TESTS" = false ]; then
            run_tests
        else
            print_warning "Skipping tests as requested"
        fi
        show_info
    else
        print_error "Deployment failed - API not responding"
        print_status "Check logs with: docker-compose logs fraud-detection-api"
        exit 1
    fi
}

# Handle script interruption
trap 'print_error "Deployment interrupted"; exit 1' INT TERM

# Run main function
main "$@"