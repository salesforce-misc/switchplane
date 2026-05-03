import { motion, useInView } from "framer-motion";
import { useRef, useEffect, useState, useCallback } from "react";

/* ------------------------------------------------------------------ */
/*  Shared helpers                                                     */
/* ------------------------------------------------------------------ */

/** Animate a dot along an SVG <path>. Renders as an SVG circle so nodes drawn after it will obscure it. */
function FlowDot({
  pathRef,
  color,
  delay,
  duration,
  loop,
  onProgress,
}: {
  pathRef: React.RefObject<SVGPathElement | null>;
  color: string;
  delay: number;
  duration: number;
  loop: boolean;
  onProgress?: (t: number) => void;
}) {
  const dotRef = useRef<SVGCircleElement>(null);
  const animationRef = useRef<number | null>(null);

  const animate = useCallback(() => {
    const path = pathRef.current;
    const dot = dotRef.current;
    if (!path || !dot) return;

    const totalLen = path.getTotalLength();
    const startTime = performance.now();

    function step(now: number) {
      const elapsed = now - startTime;
      const raw = elapsed / (duration * 1000);
      const t = loop ? raw % 1 : Math.min(raw, 1);
      const pt = path!.getPointAtLength(t * totalLen);
      if (dot) {
        dot.setAttribute("cx", String(pt.x));
        dot.setAttribute("cy", String(pt.y));
      }
      onProgress?.(t);
      if (loop || raw < 1) {
        animationRef.current = requestAnimationFrame(step);
      }
    }

    const timeout = setTimeout(() => {
      animationRef.current = requestAnimationFrame(step);
    }, delay * 1000);

    return () => {
      clearTimeout(timeout);
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
    };
  }, [pathRef, delay, duration, loop, onProgress]);

  useEffect(() => {
    return animate();
  }, [animate]);

  return (
    <circle
      ref={dotRef}
      r={5}
      fill={color}
      filter={`drop-shadow(0 0 6px ${color})`}
    />
  );
}

interface NodeDef {
  id: string;
  label: string;
  x: number;
  y: number;
  w: number;
  h: number;
}

function GlowNode({
  node,
  active,
  color,
}: {
  node: NodeDef;
  active: boolean;
  color: string;
}) {
  return (
    <g>
      <rect
        x={node.x}
        y={node.y}
        width={node.w}
        height={node.h}
        rx={6}
        fill={active ? color : "#0a1520"}
        stroke={active ? color : "#164e63"}
        strokeWidth={1.5}
      >
        {active && (
          <animate attributeName="fill-opacity" values="0.12;0.22;0.12" dur="1.2s" repeatCount="indefinite" />
        )}
      </rect>
      {!active && (
        <rect
          x={node.x}
          y={node.y}
          width={node.w}
          height={node.h}
          rx={6}
          fill="#0a1520"
          stroke="none"
        />
      )}
      <text
        x={node.x + node.w / 2}
        y={node.y + node.h / 2 + 4}
        textAnchor="middle"
        fill={active ? "#fff" : "#06b6d4"}
        fontSize={10}
        fontWeight={600}
        fontFamily="monospace"
      >
        {node.label}
      </text>
    </g>
  );
}

/* ------------------------------------------------------------------ */
/*  LLM-as-Runtime panel (left)                                        */
/* ------------------------------------------------------------------ */

function LlmAsRuntime({ active }: { active: boolean }) {
  const pathRef = useRef<SVGPathElement>(null);
  const [progress, setProgress] = useState(0);
  const [pathD, setPathD] = useState("");
  const cycleRef = useRef(0);

  const onProgress = useCallback((t: number) => {
    setProgress(t);
  }, []);

  const llm = { x: 60, y: 173, w: 80, h: 34 };
  const tools = [
    { id: "fetch", label: "fetch", y: 76 },
    { id: "analyze", label: "analyze", y: 176 },
    { id: "format", label: "format", y: 276 },
  ];
  const toolX = 190;
  const toolW = 80;
  const toolH = 28;
  const resultY = 340;

  const llmRight = llm.x + llm.w;
  const llmMid = llm.y + llm.h / 2;

  const toolPoint = (idx: number) => `${toolX},${tools[idx].y + toolH / 2}`;
  const llmPoint = `${llmRight},${llmMid}`;
  const resultPoint = `${toolX},${resultY + 14}`;

  const pathVariants = [
    `M ${llmPoint} L ${toolPoint(0)} L ${llmPoint} L ${toolPoint(1)} L ${llmPoint} L ${toolPoint(2)} L ${llmPoint} L ${resultPoint}`,
    `M ${llmPoint} L ${toolPoint(1)} L ${llmPoint} L ${toolPoint(0)} L ${llmPoint} L ${toolPoint(2)} L ${llmPoint} L ${resultPoint}`,
    `M ${llmPoint} L ${toolPoint(2)} L ${llmPoint} L ${toolPoint(0)} L ${llmPoint} L ${toolPoint(0)} L ${llmPoint} L ${resultPoint}`,
    `M ${llmPoint} L ${toolPoint(0)} L ${llmPoint} L ${toolPoint(2)} L ${llmPoint} L ${toolPoint(1)} L ${llmPoint} L ${toolPoint(1)} L ${llmPoint} L ${resultPoint}`,
    `M ${llmPoint} L ${toolPoint(1)} L ${llmPoint} L ${toolPoint(2)} L ${llmPoint} L ${resultPoint}`,
  ];

  useEffect(() => {
    if (!active) return;
    setPathD(pathVariants[0]);

    const interval = setInterval(() => {
      cycleRef.current = (cycleRef.current + 1) % pathVariants.length;
      setPathD(pathVariants[cycleRef.current]);
    }, 6000);

    return () => clearInterval(interval);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active]);

  const nodes: NodeDef[] = [
    { id: "llm", label: "LLM", x: llm.x, y: llm.y, w: llm.w, h: llm.h },
    { id: "fetch", label: "fetch", x: toolX, y: tools[0].y, w: toolW, h: toolH },
    { id: "analyze", label: "analyze", x: toolX, y: tools[1].y, w: toolW, h: toolH },
    { id: "format", label: "format", x: toolX, y: tools[2].y, w: toolW, h: toolH },
    { id: "result", label: "Result", x: toolX, y: resultY, w: toolW, h: 28 },
  ];

  const isNodeActive = (id: string): boolean => {
    if (id === "llm") {
      return (progress > 0.01 && progress < 0.08) ||
             (progress > 0.15 && progress < 0.22) ||
             (progress > 0.29 && progress < 0.36) ||
             (progress > 0.43 && progress < 0.50) ||
             (progress > 0.57 && progress < 0.64) ||
             (progress > 0.71 && progress < 0.78) ||
             (progress > 0.85 && progress < 0.92);
    }
    if (id === "result") return progress > 0.95;
    const currentPath = pathD;
    const toolIdx = tools.findIndex(t => t.id === id);
    if (toolIdx === -1) return false;
    const tp = toolPoint(toolIdx);
    const firstOccurrence = currentPath.indexOf(`L ${tp}`);
    if (firstOccurrence === -1) return false;
    const pathLen = currentPath.length;
    const relPos = firstOccurrence / pathLen;
    return Math.abs(progress - relPos) < 0.08 ||
           (progress > relPos - 0.02 && progress < relPos + 0.06);
  };

  return (
    <div className="relative">
      <svg aria-hidden="true" viewBox="0 0 300 400" className="h-auto w-full">
        <defs>
          <marker id="arrowGray" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
            <path d="M0,0 L8,3 L0,6" fill="#475569" />
          </marker>
        </defs>

        {/* Arrows: LLM → tool */}
        {tools.map((t) => (
          <line
            key={`link-${t.label}`}
            x1={llmRight}
            y1={llmMid}
            x2={toolX}
            y2={t.y + toolH / 2}
            stroke="#334155"
            strokeWidth={1.5}
            markerEnd="url(#arrowGray)"
          />
        ))}
        <line x1={llmRight} y1={llmMid} x2={toolX} y2={resultY + 14} stroke="#334155" strokeWidth={1.5} markerEnd="url(#arrowGray)" />

        {/* Flow path (invisible) */}
        <path
          ref={pathRef}
          d={pathD}
          fill="none"
          stroke="none"
        />

        {/* Animated dot */}
        {active && pathD && (
          <FlowDot pathRef={pathRef} color="#f59e0b" delay={0.3} duration={6} loop={true} onProgress={onProgress} />
        )}

        {/* Nodes with glow (rendered after dot to obscure it) */}
        {nodes.filter(n => n.id !== "llm").map((n) => (
          <GlowNode key={n.id} node={n} active={active && isNodeActive(n.id)} color="#f59e0b" />
        ))}

        {/* LLM hub node (always amber-stroked) */}
        <rect
          x={llm.x} y={llm.y} width={llm.w} height={llm.h} rx={6}
          fill={active && isNodeActive("llm") ? "#f59e0b" : "#1c1406"}
          stroke="#f59e0b"
          strokeWidth={1.5}
        >
          {active && isNodeActive("llm") && (
            <animate attributeName="fill-opacity" values="0.12;0.22;0.12" dur="1.2s" repeatCount="indefinite" />
          )}
        </rect>
        <text
          x={llm.x + llm.w / 2} y={llm.y + llm.h / 2 + 5}
          textAnchor="middle" fill={active && isNodeActive("llm") ? "#fff" : "#f59e0b"}
          fontSize={13} fontWeight={600} fontFamily="monospace"
        >
          LLM
        </text>

        {/* Failure mode annotations */}
        <text x={llm.x + llm.w / 2} y={llm.y - 12} textAnchor="middle" fill="#f59e0b" fontSize={8} fontFamily="sans-serif" opacity={0.55}>
          different order each run
        </text>
        <text x={llm.x + llm.w / 2} y={llm.y + llm.h + 16} textAnchor="middle" fill="#f59e0b" fontSize={8} fontFamily="sans-serif" opacity={0.55}>
          may skip steps
        </text>
        <text x={toolX + toolW + 8} y={tools[0].y + toolH / 2 + 4} fill="#f59e0b" fontSize={8} fontFamily="sans-serif" opacity={0.55}>
          wrong args?
        </text>
        <text x={toolX + toolW + 8} y={tools[2].y + toolH / 2 + 4} fill="#f59e0b" fontSize={8} fontFamily="sans-serif" opacity={0.55}>
          or not at all?
        </text>
        <text x={llm.x} y={llm.y + llm.h + 32} textAnchor="middle" fill="#f59e0b" fontSize={8} fontFamily="sans-serif" opacity={0.55}>
          infinite loop?
        </text>

        {/* Cost callout */}
        <text x={150} y={390} textAnchor="middle" fill="#f59e0b" fontSize={9} fontFamily="monospace" opacity={0.6}>
          unpredictable cost, untestable, non-reproducible
        </text>
      </svg>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Runtime-using-LLM panel (right — Switchplane)                      */
/* ------------------------------------------------------------------ */

function RuntimeUsingLlm({ active }: { active: boolean }) {
  const pathLeftRef = useRef<SVGPathElement>(null);
  const pathRightRef = useRef<SVGPathElement>(null);
  const [progress, setProgress] = useState(0);

  const onProgress = useCallback((t: number) => {
    setProgress(t);
  }, []);

  const nodes: NodeDef[] = [
    { id: "fetch-l",   label: "fetch_metrics",  x: 15,  y: 30,  w: 120, h: 32 },
    { id: "analyze-l", label: "analyze_metrics", x: 15,  y: 110, w: 120, h: 32 },
    { id: "fetch-r",   label: "fetch_metrics",  x: 185, y: 30,  w: 120, h: 32 },
    { id: "analyze-r", label: "analyze_metrics", x: 185, y: 110, w: 120, h: 32 },
    { id: "summarize", label: "summarize",      x: 100, y: 200, w: 120, h: 32 },
    { id: "compile",   label: "compile_report", x: 100, y: 290, w: 120, h: 32 },
  ];

  // Thresholds offset slightly so dot is visible approaching the node before it lights up
  const isNodeActive = (id: string): boolean => {
    switch (id) {
      case "fetch-l":
      case "fetch-r":   return progress >= 0.03 && progress < 0.15;
      case "analyze-l":
      case "analyze-r": return progress >= 0.21 && progress < 0.35;
      case "summarize": return progress >= 0.41 && progress < 0.7;
      case "compile":   return progress >= 0.88;
      default: return false;
    }
  };

  return (
    <div className="relative">
      <svg aria-hidden="true" viewBox="0 0 320 380" className="h-auto w-full">
        <defs>
          <marker id="arrowCyan" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
            <path d="M0,0 L8,3 L0,6" fill="#06b6d4" />
          </marker>
        </defs>

        {/* Edges (rendered behind nodes) */}
        <line x1={75} y1={62} x2={75} y2={110} stroke="#164e63" strokeWidth={1.5} markerEnd="url(#arrowCyan)" />
        <line x1={245} y1={62} x2={245} y2={110} stroke="#164e63" strokeWidth={1.5} markerEnd="url(#arrowCyan)" />
        <line x1={75} y1={142} x2={130} y2={200} stroke="#164e63" strokeWidth={1.5} markerEnd="url(#arrowCyan)" />
        <line x1={245} y1={142} x2={190} y2={200} stroke="#164e63" strokeWidth={1.5} markerEnd="url(#arrowCyan)" />
        <line x1={160} y1={232} x2={160} y2={290} stroke="#164e63" strokeWidth={1.5} markerEnd="url(#arrowCyan)" />

        {/* LLM side-call line (behind everything) */}
        <line x1={220} y1={216} x2={258} y2={206} stroke="#f59e0b" strokeWidth={1} strokeDasharray="4 3" opacity={0.6} />

        {/* Node type labels */}
        <text x={75} y={72} textAnchor="middle" fill="#06b6d4" fontSize={9} fontFamily="sans-serif" opacity={0.5}>http</text>
        <text x={75} y={151} textAnchor="middle" fill="#06b6d4" fontSize={9} fontFamily="sans-serif" opacity={0.5}>pandas</text>
        <text x={245} y={72} textAnchor="middle" fill="#06b6d4" fontSize={9} fontFamily="sans-serif" opacity={0.5}>http</text>
        <text x={245} y={151} textAnchor="middle" fill="#06b6d4" fontSize={9} fontFamily="sans-serif" opacity={0.5}>pandas</text>
        <text x={93} y={221} textAnchor="end" fill="#f59e0b" fontSize={9} fontFamily="sans-serif" opacity={0.5}>claude</text>
        <text x={93} y={311} textAnchor="end" fill="#06b6d4" fontSize={9} fontFamily="sans-serif" opacity={0.5}>format</text>

        {/* Flow paths (invisible) */}
        <path
          ref={pathLeftRef}
          d="M 75,46 L 75,126 L 160,216 L 279,206 L 160,216 L 160,306"
          fill="none"
          stroke="none"
        />
        <path
          ref={pathRightRef}
          d="M 245,46 L 245,126 L 160,216 L 279,206 L 160,216 L 160,306"
          fill="none"
          stroke="none"
        />

        {/* Animated dots (rendered before nodes so nodes obscure them) */}
        {active && (
          <>
            <FlowDot pathRef={pathLeftRef} color="#06b6d4" delay={0.3} duration={4} loop={true} onProgress={onProgress} />
            <FlowDot pathRef={pathRightRef} color="#06b6d4" delay={0.3} duration={4} loop={true} />
          </>
        )}

        {/* Nodes with glow effect (rendered after dots to obscure them) */}
        {nodes.map((n) => (
          <GlowNode key={n.id} node={n} active={active && isNodeActive(n.id)} color="#06b6d4" />
        ))}

        {/* LLM node (rendered after dots so it obscures them) */}
        <rect
          x={258} y={195} width={42} height={22} rx={4}
          fill={progress >= 0.48 && progress < 0.62 ? "#2a1a00" : "#1c1406"}
          stroke="#f59e0b"
          strokeWidth={progress >= 0.48 && progress < 0.62 ? 2 : 1}
          opacity={progress >= 0.48 && progress < 0.62 ? 1 : 0.6}
        />
        <text x={279} y={210} textAnchor="middle" fill="#f59e0b" fontSize={9} fontFamily="monospace" opacity={progress >= 0.48 && progress < 0.62 ? 1 : 0.7}>LLM</text>

        {/* Cost callout */}
        <text x={160} y={350} textAnchor="middle" fill="#06b6d4" fontSize={10} fontFamily="monospace" opacity={0.6}>
          1 LLM call (only where judgment needed)
        </text>
      </svg>
    </div>
  );
}


/* ------------------------------------------------------------------ */
/*  Main Comparison section                                            */
/* ------------------------------------------------------------------ */

function Comparison() {
  const sectionRef = useRef<HTMLDivElement>(null);
  const isInView = useInView(sectionRef, { once: false, margin: "-100px" });

  return (
    <section
      ref={sectionRef}
      className="relative mx-auto max-w-6xl px-6 py-20"
    >
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.6 }}
        className="mb-16 text-center"
      >
        <h2 className="mb-4 text-3xl font-bold text-gray-100 sm:text-4xl">
          Two approaches to agent orchestration
        </h2>
        <p className="mx-auto max-w-xl text-gray-400">
          The default pattern in agentic code is to let the LLM own the event loop.
          Switchplane inverts this: a deterministic graph defines the operational
          boundaries. LLM nodes can still reason, loop, and use tools, but only
          within the scope the graph assigns them.
        </p>
      </motion.div>

      <div className="mb-8 grid gap-8 md:grid-cols-2">
        {/* Left panel: LLM-as-Runtime */}
        <motion.div
          initial={{ opacity: 0, x: -30 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="rounded-xl border border-amber-500/20 bg-surface-light/50 p-6"
        >
          <div className="mb-6 flex items-center gap-3">
            <div className="h-3 w-3 rounded-full bg-amber-500" />
            <h3 className="text-lg font-semibold text-amber-400">
              LLM-as-Runtime
            </h3>
            <span className="ml-auto rounded-full border border-amber-500/30 px-2 py-0.5 text-xs text-amber-500/70">
              typical pattern
            </span>
          </div>
          <p className="mb-6 leading-relaxed text-gray-400">
            The LLM is the orchestrator. It decides the next step, calls tools,
            evaluates results, and loops until it thinks it is done. Control flow
            is emergent and non-deterministic.
          </p>
          <div className="relative min-h-[370px]">
            <LlmAsRuntime active={isInView} />
          </div>
        </motion.div>

        {/* Right panel: Runtime-using-LLM */}
        <motion.div
          initial={{ opacity: 0, x: 30 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="rounded-xl border border-cyan-500/20 bg-surface-light/50 p-6"
        >
          <div className="mb-6 flex items-center gap-3">
            <div className="h-3 w-3 rounded-full bg-cyan-500" />
            <h3 className="text-lg font-semibold text-cyan-400">
              Runtime-using-LLM
            </h3>
            <span className="ml-auto rounded-full border border-cyan-500/30 px-2 py-0.5 text-xs text-cyan-500/70">
              switchplane
            </span>
          </div>
          <p className="mb-6 leading-relaxed text-gray-400">
            A deterministic StateGraph defines the operational boundary. Each node
            can do anything (including full LLM reasoning loops) but the graph
            decides what runs, when, and with what scope. Control flow is explicit
            and inspectable.
          </p>
          <div className="relative min-h-[370px]">
            <RuntimeUsingLlm active={isInView} />
          </div>
        </motion.div>
      </div>

      <p className="mx-auto max-w-xl text-center text-sm text-gray-500">
        Both panels visualize the same task: a weekly ops review that fetches metrics,
        analyzes them, and produces a summary. Based on the{" "}
        <a
          href="https://github.com/salesforce-misc/switchplane/tree/main/examples/devops"
          className="text-cyan-500 underline decoration-cyan-500/30 hover:decoration-cyan-500"
        >
          examples/devops
        </a>{" "}
        example.
      </p>
    </section>
  );
}

export default Comparison;
