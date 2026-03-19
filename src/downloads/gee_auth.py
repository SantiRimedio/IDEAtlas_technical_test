"""
Google Earth Engine Authentication and Initialization for Colab
================================================================

This module handles GEE authentication and initialization in Google Colab environment.
Provides seamless authentication flow with clear error handling and status feedback.

Author: Adapted for ML inference preprocessing pipeline
Compatible with: Google Colab, Jupyter notebooks
Dependencies: earthengine-api
"""

import ee
import os
import time
from typing import Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class GEEAuthenticator:
    """
    Handles Google Earth Engine authentication and initialization in Colab.
    
    Provides robust authentication with fallback methods and clear error reporting.
    """
    
    def __init__(self, project_id: Optional[str] = None):
        """
        Initialize GEE authenticator.
        
        Parameters:
        -----------
        project_id : str, optional
            Google Cloud project ID. If None, uses default project.
        """
        self.project_id = project_id
        self.is_authenticated = False
        self.is_initialized = False
        self.auth_method = None
        
    def initialize_gee(self, max_retries: int = 3, retry_delay: int = 2) -> bool:
        """
        Initialize and authenticate Google Earth Engine.
        
        Attempts authentication with automatic retry and fallback methods.
        
        Parameters:
        -----------
        max_retries : int, default 3
            Maximum number of authentication attempts
        retry_delay : int, default 2
            Delay between retry attempts (seconds)
            
        Returns:
        --------
        bool
            True if authentication successful, False otherwise
        """
        print("🌍 Initializing Google Earth Engine...")
        
        # Check if already initialized
        if self._check_existing_auth():
            return True
        
        # Attempt authentication with retries
        for attempt in range(max_retries):
            try:
                print(f"🔐 Authentication attempt {attempt + 1}/{max_retries}")
                
                if self._attempt_authentication():
                    if self._initialize_ee():
                        print("✅ Google Earth Engine initialized successfully!")
                        self._print_auth_status()
                        return True
                        
            except Exception as e:
                print(f"❌ Authentication attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < max_retries - 1:
                    print(f"⏳ Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print("❌ All authentication attempts failed.")
                    self._print_troubleshooting_help()
                    
        return False
    
    def _check_existing_auth(self) -> bool:
        """Check if GEE is already authenticated and initialized."""
        try:
            # Try a simple GEE operation
            ee.Number(1).getInfo()
            print("✅ GEE already authenticated and initialized")
            self.is_authenticated = True
            self.is_initialized = True
            self.auth_method = "existing"
            return True
        except:
            return False
    
    def _attempt_authentication(self) -> bool:
        """
        Attempt GEE authentication using available methods.
        
        Returns:
        --------
        bool
            True if authentication successful, False otherwise
        """
        # Method 1: Try service account authentication (if available)
        if self._try_service_account_auth():
            return True
            
        # Method 2: Try user authentication
        if self._try_user_auth():
            return True
            
        return False
    
    def _try_service_account_auth(self) -> bool:
        """Attempt service account authentication."""
        try:
            # Check for service account key
            service_account_key = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
            if service_account_key and os.path.exists(service_account_key):
                print("🔑 Attempting service account authentication...")
                
                # Initialize with service account
                credentials = ee.ServiceAccountCredentials(
                    email=None,  # Auto-detect from key file
                    key_file=service_account_key
                )
                ee.Initialize(credentials)
                
                self.auth_method = "service_account"
                self.is_authenticated = True
                print("✅ Service account authentication successful")
                return True
                
        except Exception as e:
            print(f"⚠️ Service account authentication failed: {str(e)}")
            
        return False
    
    def _try_user_auth(self) -> bool:
        """Attempt user authentication."""
        try:
            print("👤 Attempting user authentication...")
            print("📝 Please follow the authentication prompts below:")
            
            # Trigger authentication flow
            ee.Authenticate()
            
            self.auth_method = "user"
            self.is_authenticated = True
            print("✅ User authentication successful")
            return True
            
        except Exception as e:
            print(f"❌ User authentication failed: {str(e)}")
            return False
    
    def _initialize_ee(self) -> bool:
        """Initialize Earth Engine after authentication."""
        try:
            if self.project_id:
                print(f"🚀 Initializing with project ID: {self.project_id}")
                ee.Initialize(project=self.project_id)
            else:
                print("🚀 Initializing with default project...")
                ee.Initialize()
                
            # Test initialization with a simple operation
            test_result = ee.Number(1).add(1).getInfo()
            if test_result == 2:
                self.is_initialized = True
                return True
            else:
                raise Exception("Initialization test failed")
                
        except Exception as e:
            print(f"❌ Earth Engine initialization failed: {str(e)}")
            return False
    
    def check_authentication_status(self) -> Dict[str, Any]:
        """
        Check current authentication status and return detailed information.
        
        Returns:
        --------
        dict
            Dictionary containing authentication status details
        """
        status = {
            'authenticated': False,
            'initialized': False,
            'auth_method': None,
            'project_id': self.project_id,
            'can_access_gee': False,
            'error': None
        }
        
        try:
            # Test basic GEE operation
            test_result = ee.Number(42).getInfo()
            
            status.update({
                'authenticated': True,
                'initialized': True,
                'auth_method': self.auth_method or 'unknown',
                'can_access_gee': True
            })
            
        except Exception as e:
            status['error'] = str(e)
            
        return status
    
    def setup_colab_environment(self) -> None:
        """
        Set up optimal GEE configuration for Colab environment.
        
        Configures timeouts, retry settings, and other Colab-specific optimizations.
        """
        print("⚙️ Setting up Colab environment optimizations...")
        
        # Set longer timeouts for Colab's potentially slower connections
        os.environ['EE_TIMEOUT'] = '300'  # 5 minutes
        os.environ['EE_MAX_RETRIES'] = '3'
        
        # Configure for Colab's memory constraints
        if 'COLAB_GPU' in os.environ:
            print("🚀 GPU detected - configuring for GPU environment")
        
        print("✅ Colab environment configured")
    
    def _print_auth_status(self) -> None:
        """Print formatted authentication status."""
        print("\n" + "="*50)
        print("🌍 GOOGLE EARTH ENGINE STATUS")
        print("="*50)
        print(f"✅ Authentication: {'SUCCESS' if self.is_authenticated else 'FAILED'}")
        print(f"✅ Initialization: {'SUCCESS' if self.is_initialized else 'FAILED'}")
        print(f"🔑 Auth Method: {self.auth_method or 'None'}")
        print(f"🏗️ Project ID: {self.project_id or 'Default'}")
        print("="*50 + "\n")
    
    def _print_troubleshooting_help(self) -> None:
        """Print troubleshooting help for authentication failures."""
        print("\n" + "🆘 TROUBLESHOOTING HELP")
        print("="*50)
        print("If authentication failed, try these steps:")
        print()
        print("1. 🔄 Restart Runtime:")
        print("   Runtime → Restart runtime")
        print()
        print("2. 🧹 Clear Authentication:")
        print("   !earthengine authenticate --quiet")
        print()
        print("3. 🔑 Manual Authentication:")
        print("   import ee")
        print("   ee.Authenticate()")
        print("   ee.Initialize()")
        print()
        print("4. 📧 Check Account Access:")
        print("   Ensure your Google account has Earth Engine access")
        print("   Visit: https://earthengine.google.com/")
        print()
        print("5. 🌐 Check Internet Connection:")
        print("   Verify stable internet connection")
        print("="*50 + "\n")


# Convenience functions for direct use
def initialize_gee(project_id: Optional[str] = None, 
                  max_retries: int = 3, 
                  setup_colab: bool = True) -> bool:
    """
    Convenience function to initialize GEE with default settings.
    
    Parameters:
    -----------
    project_id : str, optional
        Google Cloud project ID
    max_retries : int, default 3
        Maximum authentication retry attempts
    setup_colab : bool, default True
        Whether to apply Colab optimizations
        
    Returns:
    --------
    bool
        True if initialization successful
    """
    authenticator = GEEAuthenticator(project_id)
    
    if setup_colab:
        authenticator.setup_colab_environment()
    
    return authenticator.initialize_gee(max_retries=max_retries)


def check_gee_status() -> Dict[str, Any]:
    """
    Quick function to check current GEE status.
    
    Returns:
    --------
    dict
        Authentication status information
    """
    authenticator = GEEAuthenticator()
    return authenticator.check_authentication_status()


def quick_test_gee() -> bool:
    """
    Quick test to verify GEE is working properly.
    
    Returns:
    --------
    bool
        True if GEE is working, False otherwise
    """
    try:
        print("🧪 Testing GEE functionality...")
        
        # Test basic operations
        result = ee.Number(10).add(5).getInfo()
        if result != 15:
            raise Exception("Basic math operation failed")
        
        # Test image operation
        image = ee.Image("COPERNICUS/S2_SR_HARMONIZED/20230101T000000_20230101T000000_T1234")
        bands = image.bandNames().getInfo()
        
        print("✅ GEE functionality test passed")
        print(f"📊 Test results: Basic math = {result}")
        return True
        
    except Exception as e:
        print(f"❌ GEE functionality test failed: {str(e)}")
        return False


# Example usage and testing
if __name__ == "__main__":
    print("🌍 GEE Authentication Module Test")
    print("="*40)
    
    # Test initialization
    success = initialize_gee()
    
    if success:
        # Run functionality test
        quick_test_gee()
        
        # Show status
        status = check_gee_status()
        print("\n📊 Current Status:")
        for key, value in status.items():
            print(f"   {key}: {value}")
    else:
        print("❌ Initialization failed - check troubleshooting steps above")