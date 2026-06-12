// Fire once when an element first scrolls near the viewport. Used to defer plot
// fetch + render until a card is actually about to be seen, so the initial paint
// stays fast even when a tab holds several heavy plots.
import { useEffect, useRef, useState } from "react";

export function useInView<T extends HTMLElement>(rootMargin = "250px") {
  const ref = useRef<T>(null);
  const [inView, setInView] = useState(false);

  useEffect(() => {
    if (inView) return;
    const el = ref.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setInView(true);
          obs.disconnect();
        }
      },
      { rootMargin },
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, [inView, rootMargin]);

  return [ref, inView] as const;
}
