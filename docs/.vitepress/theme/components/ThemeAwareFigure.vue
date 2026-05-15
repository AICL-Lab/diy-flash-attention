<script setup lang="ts">
import { computed } from 'vue'
import { useData } from 'vitepress'

const props = defineProps<{
  light: string
  dark: string
  alt: string
  caption?: string
}>()

const { isDark } = useData()
const src = computed(() => (isDark.value ? props.dark : props.light))
</script>

<template>
  <figure class="theme-aware-figure">
    <img :src="src" :alt="alt" />
    <figcaption v-if="caption">{{ caption }}</figcaption>
  </figure>
</template>
