import type { Metadata } from "next";
import { Inter } from "next/font/google";
import Link from "next/link";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "MATTERGEN X | AI Material Discovery",
  description: "Accelerating materials science with generative AI.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${inter.className} min-h-screen flex flex-col`}>
        {/* Sticky Navigation Bar */}


        {/* Main Content Area */}
        <main className="flex-1 w-full">
           {children}
        </main>

        {/* Professional Footer */}
        <footer className="border-t border-gray-100 bg-white">
          <div className="mx-auto max-w-7xl py-8 px-4 sm:px-6 lg:px-8 flex flex-col md:flex-row items-center justify-between gap-4">
             <p className="text-xs text-gray-400">
               © {new Date().getFullYear()} Google DeepMind. Research Preview.
             </p>
             <div className="flex items-center gap-6">
                <span className="text-xs text-gray-400 font-medium">AI-driven materials discovery</span>
             </div>
          </div>
        </footer>
      </body>
    </html>
  );
}
