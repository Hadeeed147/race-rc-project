import { useEffect, useState } from 'react'

const TOAST_LIMIT = 5
const TOAST_REMOVE_DELAY = 4000

let count = 0
function genId() {
  count = (count + 1) % Number.MAX_SAFE_INTEGER
  return count.toString()
}

const listeners = []
let memoryState = { toasts: [] }

function dispatch(action) {
  memoryState = reducer(memoryState, action)
  listeners.forEach((l) => l(memoryState))
}

function reducer(state, action) {
  switch (action.type) {
    case 'ADD_TOAST':
      return { toasts: [action.toast, ...state.toasts].slice(0, TOAST_LIMIT) }
    case 'UPDATE_TOAST':
      return {
        toasts: state.toasts.map((t) => (t.id === action.toast.id ? { ...t, ...action.toast } : t)),
      }
    case 'DISMISS_TOAST':
      return {
        toasts: state.toasts.map((t) =>
          action.toastId === undefined || t.id === action.toastId ? { ...t, open: false } : t
        ),
      }
    case 'REMOVE_TOAST':
      return { toasts: state.toasts.filter((t) => t.id !== action.toastId) }
    default:
      return state
  }
}

export function toast({ title, description, variant, duration = TOAST_REMOVE_DELAY }) {
  const id = genId()
  const update = (props) => dispatch({ type: 'UPDATE_TOAST', toast: { id, ...props } })
  const dismiss = () => dispatch({ type: 'DISMISS_TOAST', toastId: id })
  dispatch({
    type: 'ADD_TOAST',
    toast: {
      id, title, description, variant, open: true,
      onOpenChange: (open) => { if (!open) dismiss() },
      duration,
    },
  })
  return { id, update, dismiss }
}

export function useToast() {
  const [state, setState] = useState(memoryState)
  useEffect(() => {
    listeners.push(setState)
    return () => {
      const i = listeners.indexOf(setState)
      if (i > -1) listeners.splice(i, 1)
    }
  }, [])
  return {
    toasts: state.toasts,
    toast,
    dismiss: (id) => dispatch({ type: 'DISMISS_TOAST', toastId: id }),
    remove: (id) => dispatch({ type: 'REMOVE_TOAST', toastId: id }),
  }
}
