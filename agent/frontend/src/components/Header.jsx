import React from "react";

export default function Header({ health }) {
  return (
    <header
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        marginBottom: 16,
      }}
    >
      <h1
        style={{
          fontWeight: 700,
          letterSpacing: 1,
          fontSize: "2.2rem",
          color: "#314570",
          margin: 0,
        }}
      >
        Breast Classification MCP
      </h1>
      <div
        className="status"
        style={{
          fontSize: "1.1rem",
          background: "#e9eef8",
          padding: "6px 16px",
          borderRadius: 8,
        }}
      >
        <span style={{ fontWeight: 500 }}>Health:</span>{" "}
        <strong style={{ color: health === "UP" ? "#2ecc40" : "#ff4136" }}>
          {health ?? "..."}
        </strong>
      </div>
    </header>
  );
}
