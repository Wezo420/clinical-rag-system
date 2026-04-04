"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "react-hot-toast";
import { useState } from "react";

export function Providers({ children }: { children: React.ReactNode }) {
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            staleTime: 60 * 1000,
            retry: 1,
            refetchOnWindowFocus: false,
          },
        },
      })
  );

  return (
    <QueryClientProvider client={queryClient}>
      {children}
      <Toaster
        position="bottom-right"
        toastOptions={{
          style: {
            background: "#0a2d57",
            color: "#e8f4fe",
            border: "1px solid rgba(124, 196, 251, 0.15)",
            borderRadius: "10px",
            fontSize: "0.875rem",
          },
          success: {
            iconTheme: { primary: "#10b981", secondary: "#e8f4fe" },
          },
          error: {
            iconTheme: { primary: "#f43f5e", secondary: "#e8f4fe" },
          },
        }}
      />
    </QueryClientProvider>
  );
}
