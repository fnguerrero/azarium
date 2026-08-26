-- ============================================================================
-- Azarium - esquema Postgres para Supabase
--
-- La pieza clave es Row Level Security: sin RLS, cualquier usuario logueado
-- podria leer las mesas de los demas simplemente cambiando el id en la query,
-- porque el frontend habla directo con la base. Con RLS activo, Postgres filtra
-- por auth.uid() en cada consulta y el aislamiento no depende del cliente.
--
-- Correr entero en el SQL Editor de Supabase.
-- ============================================================================

-- ---------------------------------------------------------------- mesas
create table if not exists public.mesas (
  id          uuid primary key default gen_random_uuid(),
  user_id     uuid not null references auth.users(id) on delete cascade,
  nombre      text not null check (length(trim(nombre)) between 1 and 120),
  rueda       text not null default 'europea'
              check (rueda in ('europea', 'francesa', 'americana')),
  descripcion text default '' check (length(descripcion) <= 500),
  creada      timestamptz not null default now(),
  actualizada timestamptz not null default now()
);

create index if not exists mesas_user_idx on public.mesas(user_id, creada desc);

-- ---------------------------------------------------------------- tiradas
-- Una fila por tirada. Es mas verboso que guardar un array, pero permite
-- insertar de a una sin reescribir el historial entero, y deja hacer los
-- conteos en la base cuando la mesa crece a decenas de miles de tiradas.
create table if not exists public.tiradas (
  id       bigint generated always as identity primary key,
  mesa_id  uuid not null references public.mesas(id) on delete cascade,
  casilla  text not null check (casilla ~ '^(0|00|[1-9]|[12][0-9]|3[0-6])$'),
  orden    integer not null,
  creada   timestamptz not null default now(),
  unique (mesa_id, orden)
);

create index if not exists tiradas_mesa_idx on public.tiradas(mesa_id, orden);

-- ---------------------------------------------------------------- RLS
alter table public.mesas   enable row level security;
alter table public.tiradas enable row level security;

-- Cada usuario ve y toca unicamente sus propias mesas.
drop policy if exists mesas_propias on public.mesas;
create policy mesas_propias on public.mesas
  for all
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);

-- Las tiradas heredan el permiso de su mesa.
drop policy if exists tiradas_propias on public.tiradas;
create policy tiradas_propias on public.tiradas
  for all
  using (exists (
    select 1 from public.mesas m
    where m.id = tiradas.mesa_id and m.user_id = auth.uid()
  ))
  with check (exists (
    select 1 from public.mesas m
    where m.id = tiradas.mesa_id and m.user_id = auth.uid()
  ));

-- ------------------------------------------------- frecuencias del lado del server
-- Con muchas tiradas conviene que el conteo lo haga Postgres y no el navegador.
-- La vista respeta RLS porque consulta las tablas con las policies activas.
create or replace view public.frecuencias as
  select t.mesa_id,
         t.casilla,
         count(*) as veces
  from public.tiradas t
  group by t.mesa_id, t.casilla;

-- ---------------------------------------------------------------- triggers
create or replace function public.tocar_mesa()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
  update public.mesas set actualizada = now()
  where id = coalesce(new.mesa_id, old.mesa_id);
  return coalesce(new, old);
end;
$$;

drop trigger if exists tiradas_tocan_mesa on public.tiradas;
create trigger tiradas_tocan_mesa
  after insert or delete on public.tiradas
  for each row execute function public.tocar_mesa();

-- ---------------------------------------------------------------- comprobacion
-- Despues de correr todo, verificar que RLS quedo activo en ambas tablas:
--   select relname, relrowsecurity from pg_class
--   where relname in ('mesas','tiradas');
-- Las dos tienen que devolver relrowsecurity = true.
