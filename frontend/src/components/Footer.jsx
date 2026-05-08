import { Github, BookOpen } from 'lucide-react'

export default function Footer() {
  return (
    <footer className="mt-12 border-t border-border bg-card/30">
      <div className="mx-auto flex w-full max-w-6xl flex-col items-start gap-2 px-4 md:px-6 py-6 md:flex-row md:items-center md:justify-between text-xs text-muted-foreground">
        <div className="flex items-center gap-2">
          <BookOpen className="h-3.5 w-3.5" />
          <span>RACE Reading Comprehension — classical ML quiz system. TF-IDF · LR · LinearSVC · NB · KMeans.</span>
        </div>
        <div className="flex items-center gap-3">
          <a
            href="https://www.kaggle.com/datasets/ankitdhiman7/race-dataset"
            target="_blank" rel="noreferrer"
            className="hover:text-foreground"
          >dataset</a>
          <span className="opacity-40">·</span>
          <a
            href="https://github.com/" target="_blank" rel="noreferrer"
            className="inline-flex items-center gap-1 hover:text-foreground"
          ><Github className="h-3 w-3" /> source</a>
        </div>
      </div>
    </footer>
  )
}
