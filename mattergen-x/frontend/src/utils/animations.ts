import { Variants } from "framer-motion";

// Configuration for consistent scientific feel
// Using 'as const' ensures the ease array is treated as a readonly tuple 
// compatible with Framer Motion's Easing type.
const TRANSITION = {
  duration: 0.2,
  ease: [0.0, 0.0, 0.2, 1] as const, 
};

// 1. Page / Section Entrance
export const FADE_IN: Variants = {
  initial: { opacity: 0 },
  animate: { opacity: 1, transition: TRANSITION },
  exit: { opacity: 0, transition: TRANSITION },
};

export const SLIDE_UP_FADE: Variants = {
  initial: { opacity: 0, y: 10 },
  animate: { opacity: 1, y: 0, transition: TRANSITION },
  exit: { opacity: 0, y: -10, transition: TRANSITION },
};

// 2. Staggered Container (for lists/grids)
export const STAGGER_CONTAINER: Variants = {
  initial: {},
  animate: {
    transition: {
      staggerChildren: 0.05,
      delayChildren: 0.05,
    },
  },
};

export const STAGGER_ITEM: Variants = {
  initial: { opacity: 0, y: 5 },
  animate: { opacity: 1, y: 0, transition: TRANSITION },
};

// 3. Interactive Element Feedback (Hover/Tap)
export const BUTTON_HOVER: Variants = {
  initial: { scale: 1 },
  hover: { 
    scale: 1.02, 
    transition: { duration: 0.15, ease: [0.0, 0.0, 0.2, 1] as const } 
  },
  tap: { 
    scale: 0.98, 
    transition: { duration: 0.1, ease: [0.0, 0.0, 0.2, 1] as const } 
  },
};

export const CARD_HOVER: Variants = {
  initial: { y: 0, boxShadow: "0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1)" },
  hover: { 
    y: -2, 
    boxShadow: "0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1)",
    transition: TRANSITION 
  },
};

// 4. Utility Class Helper for Motion Components
export const FADE_IN_ANIMATION = {
  initial: "initial",
  animate: "animate",
  exit: "exit",
  variants: FADE_IN
};

export const SLIDE_UP_ANIMATION = {
  initial: "initial",
  animate: "animate",
  exit: "exit",
  variants: SLIDE_UP_FADE
};
