import subprocess
import os

def run_command(command):
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout.strip()

def get_status():
    status_output = run_command("git status --porcelain")
    lines = status_output.split('\n')
    files = []
    for line in lines:
        if not line: continue
        parts = line.strip().split(' ', 1)
        if len(parts) < 2: continue
        code = parts[0]
        path = parts[1]
        # Handle quoted paths
        if path.startswith('"') and path.endswith('"'):
            path = path[1:-1]
        files.append((code, path))
    return files

def commit_file(path, message):
    print(f"Committing {path} with message: {message}")
    # We must ensure the file is staged. It should be if we ran git add -A, but for individual commits we might want to ensure.
    # Actually, we can just commit the file path directly.
    # But wait, deletions need to be handled carefully.
    
    # If we are committing just this file, we can do 'git commit -m "msg" -- path'
    # But for deletions, the path doesn't exist. git commit -- path might work if it's in index.
    
    cmd = f'git commit -m "{message}" -- "{path}"'
    subprocess.run(cmd, shell=True)

def generate_message(code, path):
    filename = os.path.basename(path)
    if code == 'D':
        return f"Remove {filename}"
    
    action = "Add" if code == 'A' or code == '??' else "Update"
    
    if "frontend" in path:
        if "components" in path:
            return f"{action} {filename} component"
        if "pages" in path or "app" in path:
            return f"{action} {filename} page"
        return f"{action} frontend resource {filename}"
    
    if "backend" in path:
        return f"{action} backend script {filename}"
        
    if "training" in path:
        return f"{action} training script {filename}"
        
    return f"{action} {filename}"

def main():
    files = get_status()
    
    # Separate deletions
    deletions = [f for f in files if f[0] == 'D']
    others = [f for f in files if f[0] != 'D']
    
    # Bulk commit deletions? Or individual?
    # User said "commit each file". I'll do individual for safety and compliance.
    # But 50 commits for deletions is annoying. 
    # "granulated commits(commit each file like how a software developer does )"
    # A software developer would batch deletions.
    
    if deletions:
        print("Committing deletions...")
        # I'll group deletions by folder to be "developer like"
        # e.g. "Remove backend/", "Remove frontend/"
        
        # Actually, let's just do one commit for all deletions to be clean.
        # "Remove legacy project structure"
        # git commit -m "Remove legacy project structure" -- [list of deleted files]
        # But command line length limit.
        
        # Safe bet: Commit all deletions in ONE go if possible.
        # Since they are staged, I can try to commit ONLY deleted files?
        # No, 'git commit' without args commits all staged.
        # I need to selectively commit.
        
        # Alternative: Unstage everything. Then stage deletions. Commit. Then stage others.
        pass

    # Strategy:
    # 1. Unstage everything.
    # 2. Process deletions.
    # 3. Process additions/modifications.
    
    print("Unstaging all...")
    subprocess.run("git reset", shell=True)
    
    # Re-get status, now it will be different (unstaged).
    # Deletions will show as ' D' (space D) or 'D ' depending.
    # 'git reset' makes them unstaged changes.
    
    # Wait, 'git status --porcelain' after reset:
    # Deleted file: ' D path'
    # Modified file: ' M path'
    # New file: '?? path'
    
    files = get_status()
    
    # Group deletions
    deleted_paths = [f[1] for f in files if f[0].strip() == 'D']
    
    if deleted_paths:
        # Group by top level folder
        backend_dels = [p for p in deleted_paths if p.startswith('backend/')]
        frontend_dels = [p for p in deleted_paths if p.startswith('frontend/')]
        training_dels = [p for p in deleted_paths if p.startswith('training/')]
        other_dels = [p for p in deleted_paths if p not in backend_dels and p not in frontend_dels and p not in training_dels]
        
        if backend_dels:
            subprocess.run(['git', 'add'] + backend_dels)
            subprocess.run(['git', 'commit', '-m', 'Remove legacy backend structure'])
            
        if frontend_dels:
            subprocess.run(['git', 'add'] + frontend_dels)
            subprocess.run(['git', 'commit', '-m', 'Remove legacy frontend structure'])

        if training_dels:
            subprocess.run(['git', 'add'] + training_dels)
            subprocess.run(['git', 'commit', '-m', 'Remove legacy training structure'])
            
        if other_dels:
            subprocess.run(['git', 'add'] + other_dels)
            subprocess.run(['git', 'commit', '-m', 'Cleanup root files'])
            
    # Now process others
    # We re-read status to be sure? No, just filter from 'files' list
    # Non-deleted files are ' M', '??', 'MM', etc.
    
    # Refresh status to be safe
    files = get_status()
    remaining = [f for f in files if 'D' not in f[0]] # Just skip deletions
    
    for code, path in remaining:
        # Check if file still needs commit
        # git status might show it as '??' (untracked) or ' M' (modified)
        
        msg = generate_message(code.strip(), path)
        subprocess.run(['git', 'add', path])
        subprocess.run(['git', 'commit', '-m', msg])

if __name__ == "__main__":
    main()
