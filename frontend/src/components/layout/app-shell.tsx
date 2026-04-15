import type { ReactNode } from "react";
import { Header } from "./header";

interface AppShellProps {
  sidebar: ReactNode;
  children: ReactNode;
}

export function AppShell({ sidebar, children }: AppShellProps) {
  return (
    <div className="flex flex-col h-screen">
      <Header />
      <div className="flex flex-1 overflow-hidden">
        <aside className="w-80 border-r border-border bg-surface-elevated overflow-y-auto flex-shrink-0 hidden lg:block">
          {sidebar}
        </aside>
        <main className="flex-1 overflow-y-auto p-6">{children}</main>
      </div>
    </div>
  );
}
