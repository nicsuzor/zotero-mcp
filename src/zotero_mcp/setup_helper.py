#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Setup helper for zotero-mcp.

This script provides utilities to automatically configure zotero-mcp
by finding the installed executable and updating Claude Desktop's config.
"""

import argparse
import getpass
import json
import os
import shutil
import sys
from pathlib import Path


def find_executable():
    """Find the full path to the zotero-mcp executable."""
    # Try to find the executable in the PATH
    exe_name = "zotero-mcp"
    if sys.platform == "win32":
        exe_name += ".exe"
    
    exe_path = shutil.which(exe_name)
    if exe_path:
        print(f"Found zotero-mcp in PATH at: {exe_path}")
        return exe_path
    
    # If not found in PATH, try to find it in common installation directories
    potential_paths = []
    
    # User site-packages
    import site
    for site_path in site.getsitepackages():
        potential_paths.append(Path(site_path) / "bin" / exe_name)
    
    # User's home directory
    potential_paths.append(Path.home() / ".local" / "bin" / exe_name)
    
    # Virtual environment
    if "VIRTUAL_ENV" in os.environ:
        potential_paths.append(Path(os.environ["VIRTUAL_ENV"]) / "bin" / exe_name)
    
    # Additional common locations
    if sys.platform == "darwin":  # macOS
        potential_paths.append(Path("/usr/local/bin") / exe_name)
        potential_paths.append(Path("/opt/homebrew/bin") / exe_name)
    
    for path in potential_paths:
        if path.exists() and os.access(path, os.X_OK):
            print(f"Found zotero-mcp at: {path}")
            return str(path)
    
    # If still not found, search in common directories
    print("Searching for zotero-mcp in common locations...")
    try:
        # On Unix-like systems, try using the 'find' command
        if sys.platform != 'win32':
            import subprocess
            result = subprocess.run(
                ["find", os.path.expanduser("~"), "-name", "zotero-mcp", "-type", "f", "-executable"],
                capture_output=True, text=True, timeout=10
            )
            paths = result.stdout.strip().split('\n')
            if paths and paths[0]:
                print(f"Found zotero-mcp at {paths[0]}")
                return paths[0]
    except Exception as e:
        print(f"Error searching for zotero-mcp: {e}")
    
    print("Warning: Could not find zotero-mcp executable.")
    print("Make sure zotero-mcp is installed and in your PATH.")
    return None


def find_claude_config():
    """Find Claude Desktop config file path."""
    config_paths = []
    
    # macOS
    if sys.platform == "darwin":
        # Try both old and new paths
        config_paths.append(Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json")
        config_paths.append(Path.home() / "Library" / "Application Support" / "Claude Desktop" / "claude_desktop_config.json")
    
    # Windows
    elif sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        if appdata:
            config_paths.append(Path(appdata) / "Claude" / "claude_desktop_config.json")
            config_paths.append(Path(appdata) / "Claude Desktop" / "claude_desktop_config.json")
    
    # Linux
    else:
        config_home = os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config')
        config_paths.append(Path(config_home) / "Claude" / "claude_desktop_config.json")
        config_paths.append(Path(config_home) / "Claude Desktop" / "claude_desktop_config.json")
    
    # Check all possible locations
    for path in config_paths:
        if path.exists():
            print(f"Found Claude Desktop config at: {path}")
            return path
    
    # Return the default path for the platform if not found
    # We'll use the newer "Claude Desktop" path as default
    if sys.platform == "darwin":  # macOS
        default_path = Path.home() / "Library" / "Application Support" / "Claude Desktop" / "claude_desktop_config.json"
    elif sys.platform == "win32":  # Windows
        appdata = os.environ.get("APPDATA", "")
        default_path = Path(appdata) / "Claude Desktop" / "claude_desktop_config.json"
    else:  # Linux and others
        config_home = os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config')
        default_path = Path(config_home) / "Claude Desktop" / "claude_desktop_config.json"
    
    print(f"Claude Desktop config not found. Using default path: {default_path}")
    return default_path

def setup_semantic_search(existing_semantic_config: dict = None, semantic_config_only_arg: bool = False) -> dict:
    """Interactive setup for semantic search configuration."""
    print("\n=== Semantic Search Configuration ===")

    if existing_semantic_config:
        # Display config without sensitive info
        model = existing_semantic_config.get("embedding_model", "unknown")
        name = existing_semantic_config.get("embedding_config", {}).get("model_name", "unknown")
        update_freq = existing_semantic_config.get("update_config", {}).get("update_frequency", "unknown")
        print("Found existing semantic search configuration:")
        print(f"  - Embedding model: {model}")
        print(f"  - Embedding model name: {name}")
        print(f"  - Update frequency: {update_freq}")
        print("You can keep it or change it.")
        print("If you change to a new configuration, a database rebuild is advised.")
        print("Would you like to keep your existing configuration? (y/n): ", end="")
        if input().strip().lower() in ['y', 'yes']:
            return existing_semantic_config
    
    print("Configure embedding models for semantic search over your Zotero library.")
    
    # Choose embedding model
    print("\nAvailable embedding models:")
    print("1. Default (all-MiniLM-L6-v2) - Free, runs locally")
    print("2. OpenAI - Better quality, requires API key")
    print("3. Gemini - Better quality, requires API key")
    
    while True:
        choice = input("\nChoose embedding model (1-3): ").strip()
        if choice in ["1", "2", "3"]:
            break
        print("Please enter 1, 2, or 3")
    
    config = {}
    
    if choice == "1":
        config["embedding_model"] = "default"
        print("Using default embedding model (all-MiniLM-L6-v2)")
    
    elif choice == "2":
        config["embedding_model"] = "openai"
        
        # Choose OpenAI model
        print("\nOpenAI embedding models:")
        print("1. text-embedding-3-small (recommended, faster)")
        print("2. text-embedding-3-large (higher quality, slower)")
        
        while True:
            model_choice = input("Choose OpenAI model (1-2): ").strip()
            if model_choice in ["1", "2"]:
                break
            print("Please enter 1 or 2")
        
        if model_choice == "1":
            config["embedding_config"] = {"model_name": "text-embedding-3-small"}
        else:
            config["embedding_config"] = {"model_name": "text-embedding-3-large"}
        
        # Get API key
        api_key = getpass.getpass("Enter your OpenAI API key (hidden): ").strip()
        if api_key:
            config["embedding_config"]["api_key"] = api_key
        else:
            print("Warning: No API key provided. Set OPENAI_API_KEY environment variable.")
    
    elif choice == "3":
        config["embedding_model"] = "gemini"
        
        # Choose Gemini model
        print("\nGemini embedding models:")
        print("1. models/text-embedding-004 (recommended)")
        print("2. models/gemini-embedding-exp-03-07 (experimental)")
        
        while True:
            model_choice = input("Choose Gemini model (1-2): ").strip()
            if model_choice in ["1", "2"]:
                break
            print("Please enter 1 or 2")
        
        if model_choice == "1":
            config["embedding_config"] = {"model_name": "models/text-embedding-004"}
        else:
            config["embedding_config"] = {"model_name": "models/gemini-embedding-exp-03-07"}
        
        # Get API key
        api_key = getpass.getpass("Enter your Gemini API key (hidden): ").strip()
        if api_key:
            config["embedding_config"]["api_key"] = api_key
        else:
            print("Warning: No API key provided. Set GEMINI_API_KEY environment variable.")
    
    # Configure update frequency
    print("\n=== Database Update Configuration ===")
    print("Configure how often the semantic search database is updated:")
    print("1. Manual - Update only when you run 'zotero-mcp update-db'")
    print("2. Auto - Automatically update on server startup")
    print("3. Daily - Automatically update once per day")
    print("4. Every N days - Automatically update every N days")
    
    while True:
        update_choice = input("\nChoose update frequency (1-4): ").strip()
        if update_choice in ["1", "2", "3", "4"]:
            break
        print("Please enter 1, 2, 3, or 4")
    
    update_config = {}
    
    if update_choice == "1":
        update_config = {
            "auto_update": False,
            "update_frequency": "manual"
        }
        print("Database will only be updated manually.")
    elif update_choice == "2":
        update_config = {
            "auto_update": True,
            "update_frequency": "startup"
        }
        print("Database will be updated every time the server starts.")
    elif update_choice == "3":
        update_config = {
            "auto_update": True,
            "update_frequency": "daily"
        }
        print("Database will be updated once per day.")
    elif update_choice == "4":
        while True:
            try:
                days = int(input("Enter number of days between updates: ").strip())
                if days > 0:
                    break
                print("Please enter a positive number")
            except ValueError:
                print("Please enter a valid number")
        
        update_config = {
            "auto_update": True,
            "update_frequency": f"every_{days}",
            "update_days": days
        }
        print(f"Database will be updated every {days} days.")
    
    # Configure fulltext processing
    print("\n=== Full-Text Processing Configuration ===")
    print("Enable full-text PDF processing for semantic search?")
    print("This will extract and index content from PDF attachments.")
    print("1. Yes - Enable full-text processing (recommended)")
    print("2. No - Only index metadata (faster, less comprehensive)")
    
    while True:
        fulltext_choice = input("\nEnable full-text processing? (1-2): ").strip()
        if fulltext_choice in ["1", "2"]:
            break
        print("Please enter 1 or 2")
    
    fulltext_config = {
        "enabled": fulltext_choice == "1",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "max_chunks_per_item": 50,
        "include_annotations": True
    }
    
    if fulltext_choice == "1":
        print("\nFull-text processing enabled with default settings:")
        print("- Chunk size: 1000 tokens")
        print("- Chunk overlap: 200 tokens")
        print("- Max chunks per item: 50")
        print("- Include PDF annotations: Yes")
        
        # Option to customize settings
        customize = input("\nCustomize these settings? (y/N): ").strip().lower()
        if customize in ["y", "yes"]:
            print("\n=== Full-Text Processing Advanced Settings ===")
            
            while True:
                try:
                    chunk_size = int(input(f"Chunk size in tokens ({fulltext_config['chunk_size']}): ").strip() or fulltext_config['chunk_size'])
                    if 100 <= chunk_size <= 5000:
                        fulltext_config['chunk_size'] = chunk_size
                        break
                    print("Chunk size must be between 100 and 5000 tokens")
                except ValueError:
                    print("Please enter a valid number")
            
            while True:
                try:
                    overlap = int(input(f"Chunk overlap in tokens ({fulltext_config['chunk_overlap']}): ").strip() or fulltext_config['chunk_overlap'])
                    if 0 <= overlap <= chunk_size // 2:
                        fulltext_config['chunk_overlap'] = overlap
                        break
                    print(f"Overlap must be between 0 and {chunk_size // 2} tokens")
                except ValueError:
                    print("Please enter a valid number")
            
            while True:
                try:
                    max_chunks = int(input(f"Max chunks per item ({fulltext_config['max_chunks_per_item']}): ").strip() or fulltext_config['max_chunks_per_item'])
                    if 1 <= max_chunks <= 200:
                        fulltext_config['max_chunks_per_item'] = max_chunks
                        break
                    print("Max chunks must be between 1 and 200")
                except ValueError:
                    print("Please enter a valid number")
            
            annotations = input(f"Include PDF annotations? (Y/n): ").strip().lower()
            fulltext_config['include_annotations'] = annotations not in ["n", "no"]
    else:
        print("Full-text processing disabled. Only metadata will be indexed.")
    
    config["update_config"] = update_config
    config["fulltext"] = fulltext_config
    
    return config


def save_semantic_search_config(config: dict, semantic_config_path: Path) -> bool:
    """Save semantic search configuration to file."""
    try:
        # Ensure config directory exists
        semantic_config_dir = semantic_config_path.parent
        semantic_config_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing config or create new one
        full_semantic_config = {}
        if semantic_config_path.exists():
            try:
                with open(semantic_config_path, 'r') as f:
                    full_semantic_config = json.load(f)
            except json.JSONDecodeError:
                print("Warning: Existing semantic search config file is invalid JSON, creating new one")
        
        # Add semantic search config
        full_semantic_config["semantic_search"] = config
        
        # Write config
        with open(semantic_config_path, 'w') as f:
            json.dump(full_semantic_config, f, indent=2)
        
        print(f"Semantic search configuration saved to: {semantic_config_path}")
        return True
        
    except Exception as e:
        print(f"Error saving semantic search config: {e}")
        return False
    
def load_semantic_search_config(semantic_config_path: Path) -> dict:
    """Load existing semantic search configuration."""
    if not semantic_config_path.exists():
        return {}
    
    try:
        with open(semantic_config_path, 'r') as f:
            full_semantic_config = json.load(f)
        return full_semantic_config.get("semantic_search", {})
    except json.JSONDecodeError as e:
        print(f"Warning: Could not parse config file as JSON: {e}")
        return {}
    except Exception as e:
        print(f"Warning: Could not read config file: {e}")
        return {}


def update_claude_config(config_path, zotero_mcp_path, local=True, api_key=None, library_id=None, library_type="user", semantic_config=None):
    """Update Claude Desktop config to add zotero-mcp."""
    # Create directory if it doesn't exist
    config_dir = config_path.parent
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Load existing config or create new one
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"Loaded existing config from: {config_path}")
        except json.JSONDecodeError:
            print(f"Error: Config file at {config_path} is not valid JSON. Creating new config.")
            config = {}
    else:
        print(f"Creating new config file at: {config_path}")
        config = {}
    
    # Ensure mcpServers key exists
    if "mcpServers" not in config:
        config["mcpServers"] = {}
    
    # Create environment settings based on local vs web API
    env_settings = {
        "ZOTERO_LOCAL": "true" if local else "false"
    }
    
    # Add API key and library settings for web API
    if not local:
        if api_key:
            env_settings["ZOTERO_API_KEY"] = api_key
        if library_id:
            env_settings["ZOTERO_LIBRARY_ID"] = library_id
        if library_type:
            env_settings["ZOTERO_LIBRARY_TYPE"] = library_type
    
    # Add semantic search settings if provided
    if semantic_config:
        env_settings["ZOTERO_EMBEDDING_MODEL"] = semantic_config.get("embedding_model", "default")
        
        embedding_config = semantic_config.get("embedding_config", {})
        if semantic_config.get("embedding_model") == "openai":
            if api_key := embedding_config.get("api_key"):
                env_settings["OPENAI_API_KEY"] = api_key
            if model := embedding_config.get("model_name"):
                env_settings["OPENAI_EMBEDDING_MODEL"] = model
        
        elif semantic_config.get("embedding_model") == "gemini":
            if api_key := embedding_config.get("api_key"):
                env_settings["GEMINI_API_KEY"] = api_key
            if model := embedding_config.get("model_name"):
                env_settings["GEMINI_EMBEDDING_MODEL"] = model
    
    # Add or update zotero config
    config["mcpServers"]["zotero"] = {
        "command": zotero_mcp_path,
        "env": env_settings
    }
    
    # Write updated config
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"\nSuccessfully wrote config to: {config_path}")
    except Exception as e:
        print(f"Error writing config file: {str(e)}")
        return False
    
    return config_path


def main(cli_args=None):
    """Main function to run the setup helper."""
    parser = argparse.ArgumentParser(description="Configure zotero-mcp for Claude Desktop")
    parser.add_argument("--no-local", action="store_true", help="Configure for Zotero Web API instead of local API")
    parser.add_argument("--api-key", help="Zotero API key (only needed with --no-local)")
    parser.add_argument("--library-id", help="Zotero library ID (only needed with --no-local)")
    parser.add_argument("--library-type", choices=["user", "group"], default="user", 
                        help="Zotero library type (only needed with --no-local)")
    parser.add_argument("--config-path", help="Path to Claude Desktop config file")
    parser.add_argument("--skip-semantic-search", action="store_true", 
                        help="Skip semantic search configuration")
    parser.add_argument("--semantic-config-only", action="store_true",
                        help="Only configure semantic search, skip Zotero setup")
    
    # If this is being called from CLI with existing args
    if cli_args is not None and hasattr(cli_args, 'no_local'):
        args = cli_args
        print("Using arguments passed from command line")
    else:
        # Otherwise parse from command line
        args = parser.parse_args()
        print("Parsed arguments from command line")

    # Determine config path for semantic search
    semantic_config_dir = Path.home() / ".config" / "zotero-mcp"
    semantic_config_path = semantic_config_dir / "config.json"
    existing_semantic_config = load_semantic_search_config(semantic_config_path)
    semantic_config_changed = False
    
    # Handle semantic search only configuration
    if args.semantic_config_only:
        print("Configuring semantic search only...")
        new_semantic_config = setup_semantic_search(existing_semantic_config)
        semantic_config_changed = existing_semantic_config != new_semantic_config
        # only save if semantic config changed
        if semantic_config_changed:
            if save_semantic_search_config(new_semantic_config, semantic_config_path):
                print("\nSemantic search configuration complete!")
                print(f"Configuration saved to: {semantic_config_path}")
                print("\nTo initialize the database, run: zotero-mcp update-db")
                return 0
            else:
                print("\nSemantic search configuration failed.")
                return 1
        else:
            print("\nSemantic search configuration left unchanged.")
            return 0
    
    # Find zotero-mcp executable
    exe_path = find_executable()
    if not exe_path:
        print("Error: Could not find zotero-mcp executable.")
        return 1
    print(f"Using zotero-mcp at: {exe_path}")
    
    # Find Claude Desktop config
    config_path = args.config_path
    if not config_path:
        config_path = find_claude_config()
    else:
        print(f"Using specified config path: {config_path}")
        config_path = Path(config_path)
    
    if not config_path:
        print("Error: Could not determine Claude Desktop config path.")
        return 1
    
    # Update config
    use_local = not args.no_local
    api_key = args.api_key
    library_id = args.library_id
    library_type = args.library_type
    
    # Configure semantic search if not skipped
    if not args.skip_semantic_search:
        # if there is already a semantic search configuration in the config file:
        if existing_semantic_config:
            print("\nFound an exisiting semantic search configuration in the config file.")
            print("Would you like to reconfigure semantic search? (y/n): ", end="")
        # if otherwise, slightly different message...
        else:
            print("\nWould you like to configure semantic search? (y/n): ", end="")
        # Either way:    
        if input().strip().lower() in ['y', 'yes']:
            new_semantic_config = setup_semantic_search(existing_semantic_config)
            if existing_semantic_config != new_semantic_config:
                semantic_config_changed = True
                existing_semantic_config = new_semantic_config  # Update the config to use
                save_semantic_search_config(existing_semantic_config, semantic_config_path)
    
    print("\nSetup with the following settings:")
    print(f"  Local API: {use_local}")
    if not use_local:
        print(f"  API Key: {api_key or 'Not provided'}")
        print(f"  Library ID: {library_id or 'Not provided'}")
        print(f"  Library Type: {library_type}")

    # Use the potentially updated semantic config
    semantic_config = existing_semantic_config

    # Update Claude Desktop config
    try:
        updated_config_path = update_claude_config(
            config_path, 
            exe_path, 
            local=use_local,
            api_key=api_key,
            library_id=library_id,
            library_type=library_type,
            semantic_config=semantic_config
        )
        
        if updated_config_path:
            print("\nSetup complete!")
            print("To use Zotero in Claude Desktop:")
            print("1. Restart Claude Desktop if it's running")
            print("2. In Claude, type: /tools zotero")            
            
            if semantic_config_changed:
                print("\nSemantic Search:")
                print("- Configured with", semantic_config.get("embedding_model", "default"), "embedding model")
                print("- To change the configuration, run: zotero-mcp setup --semantic-config-only")
                print("- The config file is located at: ~/.config/zotero-mcp/config.json")
                print("- You may need to rebuild your database: zotero-mcp update-db --force-rebuild")
            else:
                print("\nSemantic Search:")
                print("- To update the database, run: zotero-mcp update-db")
                print("- Use zotero_semantic_search tool in Claude for AI-powered search")
            
            if use_local:
                print("\nNote: Make sure Zotero desktop is running and the local API is enabled in preferences.")
            else:
                missing = []
                if not api_key:
                    missing.append("API key")
                if not library_id:
                    missing.append("Library ID")
                
                if missing:
                    print(f"\nWarning: The following required settings for Web API were not provided: {', '.join(missing)}")
                    print("You may need to set these as environment variables or reconfigure.")
            
            return 0
        else:
            print("\nSetup failed. See errors above.")
            return 1
    except Exception as e:
        print(f"\nSetup failed with error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())