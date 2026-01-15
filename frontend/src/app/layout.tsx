import Sidebar from '@/components/Sidebar';

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="flex h-screen bg-zinc-950 text-zinc-50 overflow-hidden font-sans">
        <Sidebar />
        <main className="flex-1 overflow-y-auto p-8 relative">
            <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_0%,rgba(16,185,129,0.05)_0%,transparent_50%)] pointer-events-none" />
            {children}
        </main>
      </body>
    </html>
  );
}
