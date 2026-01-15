import Link from 'next/link';

export default function Sidebar() {
  return (
    <aside className="w-64 bg-zinc-900 text-zinc-100 flex flex-col border-r border-zinc-800">
      <div className="p-6">
        <h1 className="text-xl font-bold tracking-tighter text-emerald-500">MATTERGEN X</h1>
        <p className="text-xs text-zinc-500 mt-1 uppercase tracking-widest font-mono">Research Portal</p>
      </div>
      
      <nav className="flex-1 mt-6 px-4 space-y-2">
        <Link href="/" className="block px-4 py-2 rounded bg-zinc-800 text-white font-medium transition-all">
          Dashboard
        </Link>
        <Link href="/analytics" className="block px-4 py-2 rounded text-zinc-400 hover:bg-zinc-800 hover:text-white transition-all">
          Analytics
        </Link>
        <Link href="/training" className="block px-4 py-2 rounded text-zinc-400 hover:bg-zinc-800 hover:text-white transition-all">
          Model Training
        </Link>
        <Link href="/settings" className="block px-4 py-2 rounded text-zinc-400 hover:bg-zinc-800 hover:text-white transition-all">
          Settings
        </Link>
      </nav>
      
      <div className="p-6 border-t border-zinc-800">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 rounded-full bg-emerald-600 flex items-center justify-center font-bold text-sm">
            AI
          </div>
          <div>
            <p className="text-sm font-medium">System Status</p>
            <p className="text-xs text-emerald-500 font-mono">● Operational</p>
          </div>
        </div>
      </div>
    </aside>
  );
}
